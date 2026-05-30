import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Stage1Trainer:
    def __init__(self, config, env, target_entropy,
                 state_encoder, action_encoder, goal_encoder, subgoal_encoder, action_decoder,
                 low_actor, low_critic_1, low_critic_2, target_low_critic_1, target_low_critic_2,
                 high_actor, target_high_actor, high_critic_1, high_critic_2,
                 target_high_critic_1, target_high_critic_2,
                 termination_critic, gia_network, replay_buffer):

        self.config = config
        self.env = env

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder          # 新增：将物理动作编码为隐动作
        self.goal_encoder = goal_encoder
        self.subgoal_encoder = subgoal_encoder
        self.action_decoder = action_decoder

        self.low_actor = low_actor.to(self.config.device)
        self.low_critic_1 = low_critic_1
        self.low_critic_2 = low_critic_2
        self.target_low_critic_1 = target_low_critic_1
        self.target_low_critic_2 = target_low_critic_2

        self.target_low_critic_1.load_state_dict(self.low_critic_1.state_dict())
        self.target_low_critic_2.load_state_dict(self.low_critic_2.state_dict())

        self.high_actor = high_actor
        self.target_high_actor = target_high_actor
        self.target_high_actor.load_state_dict(self.high_actor.state_dict())

        self.high_critic_1 = high_critic_1
        self.high_critic_2 = high_critic_2
        self.target_high_critic_1 = target_high_critic_1
        self.target_high_critic_2 = target_high_critic_2

        self.target_high_critic_1.load_state_dict(self.high_critic_1.state_dict())
        self.target_high_critic_2.load_state_dict(self.high_critic_2.state_dict())

        self.termination_critic = termination_critic
        self.gia = gia_network
        self.buffer = replay_buffer

        # ---------- 新增：状态解码器 & 前向动力学（用于辅助训练） ----------
        proprio_dim = config['proprio_dim']
        sensor_dim = config['sensor_dim']
        latent_dim = config['latent_dim']
        prim_dim = config['prim_dim']

        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proprio_dim + sensor_dim)
        ).to(self.config.device)

        self.forward_dyn = nn.Sequential(
            nn.Linear(latent_dim + prim_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        ).to(self.config.device)

        # 优化器
        self.state_encoder_optimizer = optim.Adam(state_encoder.parameters(), lr=config.lr_encoder)
        self.action_encoder_optimizer = optim.Adam(action_encoder.parameters(), lr=config.lr_encoder)   # 新增
        self.goal_encoder_optimizer = optim.Adam(goal_encoder.parameters(), lr=config.lr_encoder)
        self.subgoal_encoder_optimizer = optim.Adam(subgoal_encoder.parameters(), lr=config.lr_encoder)
        self.action_decoder_optimizer = optim.Adam(action_decoder.parameters(), lr=config.lr_decoder)

        self.state_decoder_opt = optim.Adam(self.state_decoder.parameters(), lr=config.lr_aux)
        self.forward_dyn_opt = optim.Adam(self.forward_dyn.parameters(), lr=config.lr_aux)

        self.low_actor_optimizer = optim.Adam(low_actor.parameters(), lr=config.lr_low)
        self.low_critic_1_optimizer = optim.Adam(low_critic_1.parameters(), lr=config.lr_low)
        self.low_critic_2_optimizer = optim.Adam(low_critic_2.parameters(), lr=config.lr_low)

        self.high_actor_optimizer = optim.Adam(high_actor.parameters(), lr=config.lr_high)
        self.high_critic_1_optimizer = optim.Adam(high_critic_1.parameters(), lr=config.lr_high)
        self.high_critic_2_optimizer = optim.Adam(high_critic_2.parameters(), lr=config.lr_high)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True, device=self.config.device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_alpha)
        self.target_entropy = target_entropy

        self.term_critic_opt = optim.Adam(termination_critic.parameters(), lr=config.lr_term)
        self.gia_opt = optim.Adam(gia_network.parameters(), lr=config.lr_gia)

        self.k = config.high_level_freq
        self.gamma = config.gamma
        self.tau = config.tau
        self.actor_noise = config.actor_noise
        self.noise_clip = config.noise_clip
        self.actor_delay = config.actor_delay
        self.max_actions = config.max_actions
        self.total_iterations = 0

        # 探索参数
        self.epsilon_term = config.get('epsilon_term', 0.2)          # 随机终止概率
        self.epsilon_goal_update = config.get('epsilon_goal_update', 0.05)  # 随机更新目标概率
        self.term_threshold = config.get('term_threshold', 0.5)      # 终止概率阈值

    def _calc_low_target(self, rewards, z_next, g_target, dones):
        next_a, log_prob = self.low_actor(z_next, g_target)
        entropy = -log_prob
        q1_value = self.target_low_critic_1(z_next, g_target, next_a)
        q2_value = self.target_low_critic_2(z_next, g_target, next_a)
        next_q = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_q * (1 - dones)
        return td_target

    def _calc_high_target(self, rewards, z_next, g_target, dones):
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
        # 阶段一暂时返回0
        return torch.zeros(z.shape[0], device=z.device)

    def _parse_obs(self, obs):
        # 根据环境实际格式拆分，此处假设 obs 是字典
        return obs['proprio'], obs['sensor'], obs['vision']

    def train_low_level(self, batch_size=256):
        obs, g, a_phys, r, obs_next, done = self.buffer.sample_low(batch_size)
        z = self.state_encoder(obs[0], obs[1], obs[2])
        z_next = self.state_encoder(obs_next[0], obs_next[1], obs_next[2])
        g_target = self.subgoal_encoder(g)

        # 清零优化器
        self.state_encoder_optimizer.zero_grad()
        self.action_encoder_optimizer.zero_grad()
        self.subgoal_encoder_optimizer.zero_grad()
        self.action_decoder_optimizer.zero_grad()
        self.low_critic_1_optimizer.zero_grad()
        self.low_critic_2_optimizer.zero_grad()
        self.low_actor_optimizer.zero_grad()
        self.log_alpha_optimizer.zero_grad()
        self.state_decoder_opt.zero_grad()
        self.forward_dyn_opt.zero_grad()

        # ---------- RL 损失 ----------
        td_target = self._calc_low_target(r, z_next, g_target, done)
        q1_pred = self.low_critic_1(z, g_target, a_phys)
        q2_pred = self.low_critic_2(z, g_target, a_phys)
        critic_1_loss = F.mse_loss(q1_pred, td_target)
        critic_2_loss = F.mse_loss(q2_pred, td_target)

        new_z_a, log_prob = self.low_actor(z, g_target)          # 隐动作
        new_a_phys, _ = self.action_decoder(new_z_a)            # 解码为物理动作
        entropy = -log_prob
        q1_value = self.low_critic_1(z, g_target, new_a_phys)
        q2_value = self.low_critic_2(z, g_target, new_a_phys)
        actor_loss = (-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value)).mean()
        alpha_loss = ((entropy - self.target_entropy).detach() * self.log_alpha.exp()).mean()

        # ---------- 辅助自监督损失 ----------
        # 1. 状态重构
        recon_target = torch.cat([obs[0], obs[1]], dim=-1)
        z_recon = self.state_decoder(z)
        recon_loss = F.mse_loss(z_recon, recon_target)

        # 2. 前向动态预测
        z_next_pred = self.forward_dyn(torch.cat([z, new_z_a], dim=-1))
        dyn_loss = F.mse_loss(z_next_pred, z_next)

        # 3. 动作循环一致性（编码器-解码器互为逆）
        z_a_cycle = self.action_encoder(a_phys)                 # 将真实物理动作编码为隐动作
        a_phys_recon, _ = self.action_decoder(z_a_cycle)        # 再解码回来
        cycle_loss = F.mse_loss(z_a_cycle, new_z_a.detach()) + F.mse_loss(a_phys_recon, a_phys)

        # 总损失
        total_loss = (critic_1_loss + critic_2_loss + actor_loss + alpha_loss +
                      self.config.lambda_recon * recon_loss +
                      self.config.lambda_dyn * dyn_loss +
                      self.config.lambda_cycle * cycle_loss)

        total_loss.backward()

        self.low_critic_1_optimizer.step()
        self.low_critic_2_optimizer.step()
        self.low_actor_optimizer.step()
        self.log_alpha_optimizer.step()
        self.state_encoder_optimizer.step()
        self.action_encoder_optimizer.step()
        self.subgoal_encoder_optimizer.step()
        self.action_decoder_optimizer.step()
        self.state_decoder_opt.step()
        self.forward_dyn_opt.step()

        self._soft_update(self.low_critic_1, self.target_low_critic_1)
        self._soft_update(self.low_critic_2, self.target_low_critic_2)

        return {'q_loss': critic_1_loss.item(), 'policy_loss': actor_loss.item()}

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
        """使用启发式标签预训练 GIA（阶段一）"""
        if len(self.buffer.high_buffer) < batch_size:
            return {}
        z, z_goal, g, _, _, _, _ = self.buffer.sample_high(batch_size)
        z = z.to(self.config.device)
        z_goal = z_goal.to(self.config.device)
        g_target = self.subgoal_encoder(g.to(self.config.device))

        with torch.no_grad():
            dist_to_goal = torch.norm(z - z_goal, dim=-1)
            # 启发式标签：距离小于阈值则 label=1（很近），否则 0（很远）
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
        proprio, sensor, vision = self._parse_obs(obs)
        # 编码到隐状态
        z = self.state_encoder(proprio.unsqueeze(0).to(self.config.device),
                               sensor.unsqueeze(0).to(self.config.device),
                               vision.unsqueeze(0).to(self.config.device)).squeeze(0)

        goal_state = self.env.get_goal_state()
        z_goal = self.goal_encoder(goal_state.unsqueeze(0).to(self.config.device)).squeeze(0)

        g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
        g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)

        low_transitions = []
        episode_reward = 0
        step = 0
        done = False

        while not done and step < max_steps:
            z_start = z.clone()
            subgoal_step = 0
            # 低层执行直到终止或达到最大步数
            while subgoal_step < self.k and not done:
                # 低层动作：隐动作
                z_a, _ = self.low_actor(z.unsqueeze(0), g_target.unsqueeze(0))
                a_phys, duration = self.action_decoder(z_a)
                duration_steps = max(1, int(duration.item() * 10))

                for _ in range(duration_steps):
                    next_obs, reward, done = self.env.step(a_phys.squeeze(0).cpu().numpy())
                    episode_reward += reward
                    step += 1
                    if done:
                        break

                proprio, sensor, vision = self._parse_obs(next_obs)
                z_next = self.state_encoder(proprio.unsqueeze(0).to(self.config.device),
                                            sensor.unsqueeze(0).to(self.config.device),
                                            vision.unsqueeze(0).to(self.config.device)).squeeze(0)

                # 内在奖励（子目标进度）
                r_intrinsic = -torch.norm(z_next - (z + g_target), p=2).item()

                # 存储低层经验（存 a_phys 而不是隐动作，用于循环一致性训练）
                self.buffer.add_low((proprio.cpu().numpy(), sensor.cpu().numpy(), vision.cpu().numpy()),
                                    g_target.cpu().numpy(),
                                    a_phys.squeeze(0).cpu().numpy(),
                                    r_intrinsic,
                                    (proprio.cpu().numpy(), sensor.cpu().numpy(), vision.cpu().numpy()),
                                    done)
                low_transitions.append((z.cpu().numpy(), z_a.cpu().numpy(), r_intrinsic, z_next.cpu().numpy()))
                z = z_next
                subgoal_step += 1

                # 调度器判断是否提前终止（ε-greedy）
                term_prob = self.termination_critic(z.unsqueeze(0), g_target.unsqueeze(0)).item()
                if np.random.random() < self.epsilon_term:
                    terminate = True
                else:
                    terminate = (term_prob > self.term_threshold)
                if terminate or done:
                    total_r = sum([t[2] for t in low_transitions])
                    self.buffer.add_high(z_start.cpu().numpy(),
                                         goal_state.cpu().numpy() if isinstance(goal_state, torch.Tensor) else goal_state,
                                         g.cpu().numpy(),
                                         total_r,
                                         z.cpu().numpy(),
                                         done,
                                         low_transitions)
                    # 重新规划子目标
                    g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                    g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                    low_transitions = []
                    break

            # GIA 更新最终目标（ε-greedy）
            uncertainty = self._compute_uncertainty(z.unsqueeze(0), None).item()
            p_update = self.gia(z.unsqueeze(0), z_goal.unsqueeze(0),
                                g_target.unsqueeze(0),
                                torch.tensor([uncertainty], device=self.config.device)).item()
            if np.random.random() < self.epsilon_goal_update:
                should_update = True
            else:
                should_update = (p_update > self.config.gia_threshold)
            if should_update:
                new_goal_state = self.env.request_goal()
                z_goal = self.goal_encoder(new_goal_state.unsqueeze(0).to(self.config.device)).squeeze(0)
                g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                low_transitions = []

            if done:
                break

        return episode_reward

    def train(self, total_episodes):
        for ep in range(total_episodes):
            reward = self.collect_episode()
            if len(self.buffer.low_buffer) > self.config.batch_size:
                self.train_low_level()
            if len(self.buffer.high_buffer) > self.config.batch_size:
                if ep % self.config.relabel_freq == 0:
                    self.buffer.relabel_subgoals(self.low_actor, self.k)
                self.train_high_level()
            # 定期预训练 GIA（帮助收集更有意义的更新经验）
            if ep % self.config.gia_train_freq == 0 and len(self.buffer.high_buffer) > self.config.batch_size:
                self.train_gia()
            print(f"Episode {ep}, Reward: {reward:.2f}")