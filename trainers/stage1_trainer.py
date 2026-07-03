import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from networks.utils.weight_init import weight_init


class Stage1Trainer:
    def __init__(self, config, env, target_entropy,
                 state_encoder, action_encoder, goal_encoder, subgoal_encoder, action_decoder,
                 low_actor, low_critic_1, low_critic_2, target_low_critic_1, target_low_critic_2,
                 high_actor, target_high_actor, high_critic_1, high_critic_2,
                 target_high_critic_1, target_high_critic_2,
                 termination_critic, gia_network, replay_buffer):

        self.config = config
        self.env = env
        self.use_hierarchy = config.get('use_hierarchy', True)

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.goal_encoder = goal_encoder
        self.subgoal_encoder = subgoal_encoder
        self.action_decoder = action_decoder

        self.low_actor = low_actor.to(self.config.device)
        self.target_low_actor = deepcopy(low_actor).to(self.config.device)
        self.target_low_actor.load_state_dict(self.low_actor.state_dict())

        self.low_critic_1 = low_critic_1
        self.low_critic_2 = low_critic_2
        self.target_low_critic_1 = target_low_critic_1
        self.target_low_critic_2 = target_low_critic_2

        self.target_low_critic_1.load_state_dict(self.low_critic_1.state_dict())
        self.target_low_critic_2.load_state_dict(self.low_critic_2.state_dict())

        self.high_actor = high_actor
        self.target_high_actor = target_high_actor
        if self.use_hierarchy:
            self.target_high_actor.load_state_dict(self.high_actor.state_dict())

        self.high_critic_1 = high_critic_1
        self.high_critic_2 = high_critic_2
        self.target_high_critic_1 = target_high_critic_1
        self.target_high_critic_2 = target_high_critic_2

        if self.use_hierarchy:
            self.target_high_critic_1.load_state_dict(self.high_critic_1.state_dict())
            self.target_high_critic_2.load_state_dict(self.high_critic_2.state_dict())

        self.termination_critic = termination_critic
        self.gia = gia_network
        self.buffer = replay_buffer

        proprio_dim = config['proprio_dim']
        sensor_dim = config['sensor_dim']
        latent_dim = config['latent_dim']
        prim_dim = config['prim_dim']

        def _build_mlp(in_dim, out_dim):
            net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, out_dim),
            )
            net.apply(weight_init)
            return net

        self.state_decoder = _build_mlp(latent_dim, proprio_dim + sensor_dim).to(self.config.device)
        self.forward_dyn = _build_mlp(latent_dim + prim_dim, latent_dim).to(self.config.device)
        self.inverse_dyn = _build_mlp(latent_dim * 2, prim_dim).to(self.config.device)

        # 优化器
        self.state_encoder_optimizer = optim.Adam(state_encoder.parameters(), lr=config.lr_encoder)
        self.action_encoder_optimizer = optim.Adam(action_encoder.parameters(), lr=config.lr_encoder)
        self.action_decoder_optimizer = optim.Adam(action_decoder.parameters(), lr=config.lr_decoder)

        if self.use_hierarchy:
            self.goal_encoder_optimizer = optim.Adam(goal_encoder.parameters(), lr=config.lr_encoder)
            self.subgoal_encoder_optimizer = optim.Adam(subgoal_encoder.parameters(), lr=config.lr_encoder)

        self.state_decoder_opt = optim.Adam(self.state_decoder.parameters(), lr=config.lr_aux)
        self.forward_dyn_opt = optim.Adam(self.forward_dyn.parameters(), lr=config.lr_aux)
        self.inverse_dyn_opt = optim.Adam(self.inverse_dyn.parameters(), lr=config.lr_aux)

        self.low_actor_optimizer = optim.Adam(low_actor.parameters(), lr=config.lr_low)
        self.low_critic_1_optimizer = optim.Adam(low_critic_1.parameters(), lr=config.lr_low)
        self.low_critic_2_optimizer = optim.Adam(low_critic_2.parameters(), lr=config.lr_low)

        if self.use_hierarchy:
            self.high_actor_optimizer = optim.Adam(high_actor.parameters(), lr=config.lr_high)
            self.high_critic_1_optimizer = optim.Adam(high_critic_1.parameters(), lr=config.lr_high)
            self.high_critic_2_optimizer = optim.Adam(high_critic_2.parameters(), lr=config.lr_high)
            self.term_critic_opt = optim.Adam(termination_critic.parameters(), lr=config.lr_term)
            self.gia_opt = optim.Adam(gia_network.parameters(), lr=config.lr_gia)

        self.k = config.high_level_freq
        self.gamma = config.gamma
        self.tau = config.tau
        self.actor_noise = config.actor_noise
        self.noise_clip = config.noise_clip
        self.actor_delay = config.get('policy_delay', 2)
        self.max_actions = config.max_actions
        self._low_step = 0
        self.total_iterations = 0
        self.epsilon_term = config.get('epsilon_term', 0.2)
        self.epsilon_goal_update = config.get('epsilon_goal_update', 0.05)
        self.term_threshold = config.get('term_threshold', 0.5)

    def load_checkpoint(self, path, resume_ep=0):
        ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt['state_encoder'])
        self.action_encoder.load_state_dict(ckpt['action_encoder'])
        self.action_decoder.load_state_dict(ckpt['action_decoder'])
        self.low_actor.load_state_dict(ckpt['low_actor'])
        self.target_low_actor.load_state_dict(ckpt['low_actor'])
        self.low_critic_1.load_state_dict(ckpt['low_critic_1'])
        self.low_critic_2.load_state_dict(ckpt['low_critic_2'])
        self.target_low_critic_1.load_state_dict(ckpt['low_critic_1'])
        self.target_low_critic_2.load_state_dict(ckpt['low_critic_2'])
        if self.use_hierarchy:
            if 'goal_encoder' in ckpt:
                self.goal_encoder.load_state_dict(ckpt['goal_encoder'])
                self.subgoal_encoder.load_state_dict(ckpt['subgoal_encoder'])
                self.high_actor.load_state_dict(ckpt['high_actor'])
                self.target_high_actor.load_state_dict(ckpt['high_actor'])
                self.high_critic_1.load_state_dict(ckpt['high_critic_1'])
                self.high_critic_2.load_state_dict(ckpt['high_critic_2'])
                self.target_high_critic_1.load_state_dict(ckpt['high_critic_1'])
                self.target_high_critic_2.load_state_dict(ckpt['high_critic_2'])
                self.termination_critic.load_state_dict(ckpt['termination_critic'])
                self.gia.load_state_dict(ckpt['gia'])
        if 'optimizer_state' in ckpt:
            opt = ckpt['optimizer_state']
            self.low_actor_optimizer.load_state_dict(opt['low_actor'])
            self.low_critic_1_optimizer.load_state_dict(opt['low_critic_1'])
            self.low_critic_2_optimizer.load_state_dict(opt['low_critic_2'])
            self.state_encoder_optimizer.load_state_dict(opt['state_encoder'])
            self.action_encoder_optimizer.load_state_dict(opt['action_encoder'])
            self.action_decoder_optimizer.load_state_dict(opt['action_decoder'])
            self.state_decoder_opt.load_state_dict(opt['state_decoder'])
            self.forward_dyn_opt.load_state_dict(opt['forward_dyn'])
            self.inverse_dyn_opt.load_state_dict(opt['inverse_dyn'])
        start_ep = resume_ep if resume_ep > 0 else ckpt.get('ep', 0)
        print(f"[resume] 从 {path} 加载成功，续接 ep={start_ep}")
        return start_ep

    def _calc_high_target(self, rewards, z_next, g_target, dones):
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        noise = (torch.randn_like(g_target) * self.actor_noise).clamp(-self.noise_clip, self.noise_clip)
        next_g = (self.target_high_actor(z_next, g_target) + noise).clamp(-self.max_actions, self.max_actions)
        q1_value = self.target_high_critic_1(z_next, g_target, next_g)
        q2_value = self.target_high_critic_2(z_next, g_target, next_g)
        td_target = rewards + self.gamma * (1 - dones) * torch.min(q1_value, q2_value)
        return td_target

    def _soft_update(self, source, target):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _compute_uncertainty(self, z, a):
        return torch.zeros(z.shape[0], device=z.device)

    def _parse_obs(self, obs):
        proprio = torch.from_numpy(obs['proprio']).float().to(self.config.device)
        sensor = torch.from_numpy(obs['sensor']).float().to(self.config.device)
        vision = torch.from_numpy(obs['vision']).float().to(self.config.device)
        return proprio, sensor, vision

    def update_dyn_cons(self, obs, g, a_prim, obs_next):
        proprio = obs[0]
        s_z = self.state_encoder(obs[0], obs[1], obs[2])
        s_z_next = self.state_encoder(obs_next[0], obs_next[1], obs_next[2])

        a_phys_buf = self.action_decoder(proprio, a_prim)
        a_z = self.action_encoder(proprio, a_phys_buf)

        s_z_next_pred = self.forward_dyn(torch.cat([s_z, a_z], dim=-1))
        loss_forward = F.mse_loss(s_z_next_pred, s_z_next)

        a_z_inv_pred = self.inverse_dyn(torch.cat([s_z, s_z_next], dim=-1))
        loss_backward = F.mse_loss(a_z_inv_pred, a_z)

        recon_target = torch.cat([obs[0], obs[1]], dim=-1)
        s_recon = self.state_decoder(s_z)
        loss_state_recon = F.mse_loss(s_recon, recon_target)

        a_phys_recon = self.action_decoder(proprio, a_z)
        loss_action_recon = F.mse_loss(a_phys_recon, a_phys_buf.detach())

        loss = loss_forward + loss_backward * 10 + loss_state_recon + loss_action_recon

        self.state_encoder_optimizer.zero_grad()
        self.state_decoder_opt.zero_grad()
        self.action_encoder_optimizer.zero_grad()
        self.action_decoder_optimizer.zero_grad()
        self.forward_dyn_opt.zero_grad()
        self.inverse_dyn_opt.zero_grad()

        loss.backward()

        self.state_encoder_optimizer.step()
        self.state_decoder_opt.step()
        self.action_encoder_optimizer.step()
        self.action_decoder_optimizer.step()
        self.forward_dyn_opt.step()
        self.inverse_dyn_opt.step()

        return {
            'dyn_fwd': loss_forward.item(),
            'dyn_inv': loss_backward.item(),
            'rec_state': loss_state_recon.item(),
            'rec_act': loss_action_recon.item(),
        }

    def update_actor_critic(self, obs, g, a_prim, r, obs_next, done):
        proprio, proprio_next = obs[0], obs_next[0]
        s_z = self.state_encoder(obs[0], obs[1], obs[2])
        s_z_next = self.state_encoder(obs_next[0], obs_next[1], obs_next[2])
        g_target = self.subgoal_encoder(g) if self.use_hierarchy else g

        # 将 buffer 中的隐动作解码为物理动作，用于 critic 评估
        a_phys_buf = self.action_decoder(proprio, a_prim)

        # ========== Critic 更新（TD3，每步） ==========
        with torch.no_grad():
            a_z_next = self.target_low_actor(s_z_next, g_target)[0]
            a_s_next = self.action_decoder(proprio_next, a_z_next)
            noise = (torch.randn_like(a_s_next) * self.actor_noise).clamp(-self.noise_clip, self.noise_clip)
            a_s_next_noisy = (a_s_next + noise).clamp(-self.max_actions, self.max_actions)
            q1_next = self.target_low_critic_1(s_z_next, g_target, a_s_next_noisy)
            q2_next = self.target_low_critic_2(s_z_next, g_target, a_s_next_noisy)
            q_next = torch.min(q1_next, q2_next)
            td_target = r.view(-1, 1) + self.gamma * (1 - done.view(-1, 1)) * q_next

        q1 = self.low_critic_1(s_z.detach(), g_target, a_phys_buf.detach())
        q2 = self.low_critic_2(s_z.detach(), g_target, a_phys_buf.detach())
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.low_critic_1_optimizer.zero_grad()
        self.low_critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.low_critic_1_optimizer.step()
        self.low_critic_2_optimizer.step()

        # ========== Actor 更新（延迟更新） ==========
        self._low_step += 1
        if self._low_step % self.actor_delay == 0:
            a_z_pred, _ = self.low_actor(s_z, g_target)
            a_s_pred = self.action_decoder(proprio, a_z_pred)
            actor_loss = -self.low_critic_1(s_z.detach(), g_target, a_s_pred).mean()

            self.low_actor_optimizer.zero_grad()
            self.state_encoder_optimizer.zero_grad()
            self.action_decoder_optimizer.zero_grad()
            if self.use_hierarchy:
                self.subgoal_encoder_optimizer.zero_grad()
                self.goal_encoder_optimizer.zero_grad()

            actor_loss.backward()

            self.low_actor_optimizer.step()
            self.state_encoder_optimizer.step()
            self.action_decoder_optimizer.step()
            if self.use_hierarchy:
                self.subgoal_encoder_optimizer.step()
                self.goal_encoder_optimizer.step()

            # 软更新目标网络
            self._soft_update(self.low_actor, self.target_low_actor)
            self._soft_update(self.low_critic_1, self.target_low_critic_1)
            self._soft_update(self.low_critic_2, self.target_low_critic_2)

            actor_loss_val = actor_loss.item()
        else:
            actor_loss_val = 0.0

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_val,
        }

    def train_low_level(self, batch_size=256):
        obs, g, a_prim, r, obs_next, done = self.buffer.sample_low(batch_size)
        obs = tuple(o.to(self.config.device) for o in obs)
        obs_next = tuple(o.to(self.config.device) for o in obs_next)
        g = g.to(self.config.device)
        a_prim = a_prim.to(self.config.device)
        r = r.to(self.config.device)
        done = done.to(self.config.device)

        dyn_metrics = self.update_dyn_cons(obs, g, a_prim, obs_next)
        ac_metrics = self.update_actor_critic(obs, g, a_prim, r, obs_next, done)

        return {
            'task_loss': ac_metrics['actor_loss'],
            'dyn_loss': dyn_metrics['dyn_fwd'] + dyn_metrics['dyn_inv'],
            'rec_loss': dyn_metrics['rec_state'] + dyn_metrics['rec_act'],
            'critic_loss': ac_metrics['critic_loss'],
        }

    def train_high_level(self, batch_size=128):
        self.total_iterations += 1
        z, goal_state, g, r, z_next, dones, _ = self.buffer.sample_high(batch_size)
        z_goal = self.goal_encoder(goal_state)

        self.state_encoder_optimizer.zero_grad()
        self.high_critic_1_optimizer.zero_grad()
        self.high_critic_2_optimizer.zero_grad()
        self.high_actor_optimizer.zero_grad()

        td_target = self._calc_high_target(r, z_next, g, dones)
        q1_pred = self.high_critic_1(z, z_goal, g)
        q2_pred = self.high_critic_2(z, z_goal, g)
        critic_1_loss = F.mse_loss(q1_pred, td_target)
        critic_2_loss = F.mse_loss(q2_pred, td_target)

        critic_1_loss.backward()
        critic_2_loss.backward()
        self.high_critic_1_optimizer.step()
        self.high_critic_2_optimizer.step()

        if self.total_iterations % self.config.policy_delay == 0:
            new_g = self.high_actor(z, z_goal)
            actor_loss = -self.high_critic_1(z, z_goal, new_g).mean()
            actor_loss.backward()
            self.high_actor_optimizer.step()
            self._soft_update(self.high_actor, self.target_high_actor)
            self._soft_update(self.high_critic_1, self.target_high_critic_1)
            self._soft_update(self.high_critic_2, self.target_high_critic_2)

        return {'high_q_loss': critic_1_loss.item()}

    def train_gia(self, batch_size=128):
        if len(self.buffer.high_buffer) < batch_size:
            return {}
        z, z_goal, g, _, _, _, _ = self.buffer.sample_high(batch_size)
        z = z.to(self.config.device)
        z_goal = z_goal.to(self.config.device)
        g_target = self.subgoal_encoder(g.to(self.config.device))

        with torch.no_grad():
            dist_to_goal = torch.norm(z - z_goal, dim=-1)
            label = (dist_to_goal < self.config.gia_dist_threshold).float()

        uncertainty = torch.zeros(batch_size, device=z.device)
        p = self.gia(z, z_goal, g_target, uncertainty)
        loss = F.binary_cross_entropy(p, label)

        self.gia_opt.zero_grad()
        loss.backward()
        self.gia_opt.step()
        return {'gia_loss': loss.item()}

    def collect_episode(self, max_steps=1000):
        obs = self.env.reset()
        n_agents = obs['proprio'].shape[0]
        proprio, sensor, vision = self._parse_obs(obs)
        self.state_encoder.eval()
        self.action_decoder.eval()
        z = self.state_encoder(proprio, sensor, vision)

        if not self.use_hierarchy:
            g_target = torch.zeros_like(z)
            episode_rewards = np.zeros(n_agents)
            dones = np.zeros(n_agents, dtype=bool)
            total_steps = 0
            while not np.all(dones) and total_steps < max_steps:
                z_a, _ = self.low_actor(z, g_target)
                a_phys = self.action_decoder(proprio, z_a)
                cur_obs = obs
                cur_z_a = z_a.detach()
                next_obs, reward, done = self.env.step(a_phys.detach().cpu().numpy())
                reward = np.asarray(reward, dtype=np.float32)
                done = np.asarray(done, dtype=bool)
                for i in range(n_agents):
                    if not dones[i]:
                        self.buffer.add_low(
                            (cur_obs['proprio'][i], cur_obs['sensor'][i], cur_obs['vision'][i]),
                            g_target[i].detach().cpu().numpy(),
                            cur_z_a[i].cpu().numpy(),
                            0.0,
                            (next_obs['proprio'][i], next_obs['sensor'][i], next_obs['vision'][i]),
                            done[i],
                        )
                        episode_rewards[i] += reward[i]
                dones = np.logical_or(dones, done)
                obs = next_obs
                total_steps += 1
                proprio, sensor, vision = self._parse_obs(obs)
                z = self.state_encoder(proprio, sensor, vision)
                if np.all(dones):
                    break
            self.state_encoder.train()
            self.action_decoder.train()
            return float(np.mean(episode_rewards)), float(np.std(episode_rewards))

        goal_state = torch.from_numpy(self.env.get_goal_state()).float().to(self.config.device)
        z_goal = self.goal_encoder(goal_state.unsqueeze(0)).squeeze(0)

        g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
        g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)

        low_transitions = []
        episode_reward = 0
        step = 0
        done = False

        while not done and step < max_steps:
            z_start = z.clone()
            subgoal_step = 0
            while subgoal_step < self.k and not done:
                z_a, _ = self.low_actor(z.unsqueeze(0), g_target.unsqueeze(0))
                a_phys = self.action_decoder(proprio.unsqueeze(0), z_a)
                next_obs, reward, done = self.env.step(a_phys.detach().squeeze(0).cpu().numpy())
                episode_reward += reward
                step += 1

                proprio, sensor, vision = self._parse_obs(next_obs)
                z_next = self.state_encoder(proprio.unsqueeze(0).to(self.config.device),
                                            sensor.unsqueeze(0).to(self.config.device),
                                            vision.unsqueeze(0).to(self.config.device)).squeeze(0)
                r_intrinsic = -torch.norm(z_next - (z + g_target), p=2).item()

                self.buffer.add_low((proprio.cpu().numpy(), sensor.cpu().numpy(), vision.cpu().numpy()),
                                    g_target.detach().cpu().numpy(),
                                    z_a.detach().squeeze(0).cpu().numpy(),
                                    r_intrinsic,
                                    (proprio.cpu().numpy(), sensor.cpu().numpy(), vision.cpu().numpy()),
                                    done)
                low_transitions.append((z.detach().cpu().numpy(), z_a.detach().cpu().numpy(), r_intrinsic, z_next.detach().cpu().numpy()))
                z = z_next
                subgoal_step += 1

                term_prob = self.termination_critic(z.unsqueeze(0), g_target.unsqueeze(0)).item()
                if np.random.random() < self.epsilon_term:
                    terminate = True
                else:
                    terminate = (term_prob > self.term_threshold)
                if terminate or done:
                    total_r = sum([t[2] for t in low_transitions])
                    self.buffer.add_high(z_start.detach().cpu().numpy(),
                                         goal_state.detach().cpu().numpy(),
                                         g.detach().cpu().numpy(), total_r,
                                         z.detach().cpu().numpy(), done, low_transitions)
                    g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                    g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                    low_transitions = []
                    break

            uncertainty = self._compute_uncertainty(z.unsqueeze(0), None).item()
            p_update = self.gia(z.unsqueeze(0), z_goal.unsqueeze(0),
                                g_target.unsqueeze(0),
                                torch.tensor([uncertainty], device=self.config.device)).item()
            if np.random.random() < self.epsilon_goal_update:
                should_update = True
            else:
                should_update = (p_update > self.config.gia_threshold)
            if should_update:
                new_goal_state = torch.from_numpy(self.env.request_goal()).float().to(self.config.device)
                z_goal = self.goal_encoder(new_goal_state.unsqueeze(0)).squeeze(0)
                g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                low_transitions = []

            if done:
                break

        return float(episode_reward), 0.0

    def train(self, total_episodes, start_ep=0):
        from pathlib import Path
        for ep in range(start_ep, start_ep + total_episodes):
            reward, reward_std = self.collect_episode()
            loss_str = ''
            if len(self.buffer.low_buffer) > self.config.batch_size:
                losses = self.train_low_level()
                loss_str = ' | '.join(f'{k}={v:.4f}' for k, v in losses.items())
            if self.use_hierarchy and len(self.buffer.high_buffer) > self.config.batch_size:
                if ep % self.config.relabel_freq == 0:
                    self.buffer.relabel_subgoals(self.low_actor, self.k)
                self.train_high_level()
            if self.use_hierarchy and ep % self.config.gia_train_freq == 0 and len(self.buffer.high_buffer) > self.config.batch_size:
                self.train_gia()

            if ep % self.config.save_freq == 0:
                save_dir = Path(self.config.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                tag = 'hier' if self.use_hierarchy else 'nohier'
                base_ckpt = {
                    'ep': ep,
                    'state_encoder': self.state_encoder.state_dict(),
                    'action_encoder': self.action_encoder.state_dict(),
                    'action_decoder': self.action_decoder.state_dict(),
                    'low_actor': self.low_actor.state_dict(),
                    'low_critic_1': self.low_critic_1.state_dict(),
                    'low_critic_2': self.low_critic_2.state_dict(),
                    'optimizer_state': {
                        'low_actor': self.low_actor_optimizer.state_dict(),
                        'low_critic_1': self.low_critic_1_optimizer.state_dict(),
                        'low_critic_2': self.low_critic_2_optimizer.state_dict(),
                        'state_encoder': self.state_encoder_optimizer.state_dict(),
                        'action_encoder': self.action_encoder_optimizer.state_dict(),
                        'action_decoder': self.action_decoder_optimizer.state_dict(),
                        'state_decoder': self.state_decoder_opt.state_dict(),
                        'forward_dyn': self.forward_dyn_opt.state_dict(),
                        'inverse_dyn': self.inverse_dyn_opt.state_dict(),
                    },
                }
                if self.use_hierarchy:
                    base_ckpt['goal_encoder'] = self.goal_encoder.state_dict()
                    base_ckpt['subgoal_encoder'] = self.subgoal_encoder.state_dict()
                    base_ckpt['high_actor'] = self.high_actor.state_dict()
                    base_ckpt['high_critic_1'] = self.high_critic_1.state_dict()
                    base_ckpt['high_critic_2'] = self.high_critic_2.state_dict()
                    base_ckpt['termination_critic'] = self.termination_critic.state_dict()
                    base_ckpt['gia'] = self.gia.state_dict()
                torch.save(base_ckpt, save_dir / f'stage1_{tag}_ep{ep}.pt')

            loss_part = f' | {loss_str}' if loss_str else ''
            print(f"Episode {ep}/{start_ep + total_episodes - 1}, Reward: {reward:.2f} \u00b1 {reward_std:.2f}{loss_part}")
