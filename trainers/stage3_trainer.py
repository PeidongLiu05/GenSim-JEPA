import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from copy import deepcopy
from pathlib import Path


class Stage3Trainer:
    """
    阶段三：基于 LeWorldModel 的内在奖励 + 自监督 GIA + 调度器

    使用阶段二的世界模型为策略提供额外内在奖励，
    并使用世界模型生成自监督标签来训练 GIA。
    """

    def __init__(self, config, env,
                 state_encoder,  # 阶段二训练好的，冻结
                 lewm_predictor,  # 阶段二训练好的，冻结
                 goal_encoder,  # 可微调
                 subgoal_encoder,  # 可微调
                 action_decoder,  # 可微调
                 low_actor,  # 低层策略 (SAC)
                 low_critic_1, low_critic_2,
                 high_actor,  # 高层策略 (TD3)
                 high_critic_1, high_critic_2,
                 termination_critic,  # 调度器
                 gia_network,  # GIA 网络 (含距离预测头)
                 replay_buffer):

        self.config = config
        self.env = env
        self.device = config.device

        # ----- 冻结世界模型 -----
        self.state_encoder = state_encoder.to(self.device)
        self.state_encoder.eval()
        for p in self.state_encoder.parameters():
            p.requires_grad = False

        self.lewm_predictor = lewm_predictor.to(self.device)
        self.lewm_predictor.eval()
        for p in self.lewm_predictor.parameters():
            p.requires_grad = False

        # ----- 可微调模块 -----
        self.goal_encoder = goal_encoder.to(self.device)
        self.subgoal_encoder = subgoal_encoder.to(self.device)
        self.action_decoder = action_decoder.to(self.device)

        self.low_actor = low_actor.to(self.device)
        self.low_critic_1 = low_critic_1.to(self.device)
        self.low_critic_2 = low_critic_2.to(self.device)
        # 为低层创建目标网络（用于 SAC）
        self.target_low_critic_1 = deepcopy(low_critic_1).to(self.device)
        self.target_low_critic_2 = deepcopy(low_critic_2).to(self.device)
        for p in self.target_low_critic_1.parameters(): p.requires_grad = False
        for p in self.target_low_critic_2.parameters(): p.requires_grad = False

        self.high_actor = high_actor.to(self.device)
        self.high_critic_1 = high_critic_1.to(self.device)
        self.high_critic_2 = high_critic_2.to(self.device)
        self.target_high_actor = deepcopy(high_actor).to(self.device)
        self.target_high_critic_1 = deepcopy(high_critic_1).to(self.device)
        self.target_high_critic_2 = deepcopy(high_critic_2).to(self.device)
        for p in self.target_high_actor.parameters(): p.requires_grad = False
        for p in self.target_high_critic_1.parameters(): p.requires_grad = False
        for p in self.target_high_critic_2.parameters(): p.requires_grad = False

        self.termination_critic = termination_critic.to(self.device)
        self.gia = gia_network.to(self.device)

        self.buffer = replay_buffer

        # ----- 优化器 -----
        # 低层 SAC 优化器
        self.low_actor_opt = torch.optim.Adam(self.low_actor.parameters(), lr=config.lr_low)
        self.low_critic1_opt = torch.optim.Adam(self.low_critic_1.parameters(), lr=config.lr_low)
        self.low_critic2_opt = torch.optim.Adam(self.low_critic_2.parameters(), lr=config.lr_low)
        self.log_alpha = torch.tensor(np.log(0.01), device=self.device, requires_grad=True)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
        self.target_entropy = -config.action_dim

        # 高层 TD3 优化器
        self.high_actor_opt = torch.optim.Adam(self.high_actor.parameters(), lr=config.lr_high)
        self.high_critic1_opt = torch.optim.Adam(self.high_critic_1.parameters(), lr=config.lr_high)
        self.high_critic2_opt = torch.optim.Adam(self.high_critic_2.parameters(), lr=config.lr_high)

        # 调度器和 GIA 优化器
        self.term_opt = torch.optim.Adam(self.termination_critic.parameters(), lr=config.lr_term)
        self.gia_opt = torch.optim.Adam(self.gia.parameters(), lr=config.lr_gia)

        # 辅助模块优化器
        self.aux_opt = torch.optim.Adam(
            list(self.goal_encoder.parameters()) +
            list(self.subgoal_encoder.parameters()) +
            list(self.action_decoder.parameters()),
            lr=config.lr_aux
        )

    def load_checkpoint(self, path, resume_ep=0):
        """从 stage3_ep*.pt 加载网络权重和优化器状态。返回续接起始 ep。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt['state_encoder'])
        self.state_encoder.eval()
        self.low_actor.load_state_dict(ckpt['low_actor'])
        self.action_decoder.load_state_dict(ckpt['action_decoder'])
        self.goal_encoder.load_state_dict(ckpt['goal_encoder'])
        self.subgoal_encoder.load_state_dict(ckpt['subgoal_encoder'])
        self.high_actor.load_state_dict(ckpt['high_actor'])
        self.gia.load_state_dict(ckpt['gia'])
        self.termination_critic.load_state_dict(ckpt['termination_critic'])
        if 'optimizer_state' in ckpt:
            opt = ckpt['optimizer_state']
            self.low_actor_opt.load_state_dict(opt['low_actor'])
            self.low_critic_1_opt.load_state_dict(opt['low_critic_1'])
            self.low_critic_2_opt.load_state_dict(opt['low_critic_2'])
            self.high_actor_opt.load_state_dict(opt['high_actor'])
            self.high_critic1_opt.load_state_dict(opt['high_critic_1'])
            self.high_critic2_opt.load_state_dict(opt['high_critic_2'])
            self.aux_opt.load_state_dict(opt['aux'])
        start_ep = resume_ep if resume_ep > 0 else ckpt.get('ep', 0)
        print(f"[resume] 从 {path} 加载成功，续接 ep={start_ep}")
        return start_ep

        # 超参数
        self.gamma = config.gamma
        self.tau = config.tau
        self.k = config.high_level_freq
        self.actor_noise = config.actor_noise
        self.noise_clip = config.noise_clip
        self.policy_delay = config.policy_delay
        self.max_actions = config.max_actions          # TD3 动作裁剪范围
        self.update_counter = 0

        # 内在奖励权重
        self.lambda_wm = config.lambda_wm

        # GIA 自监督参数
        self.gia_rollout_horizon = config.gia_rollout_horizon

    # ------------------- 辅助函数 -------------------
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

    def _parse_obs(self, obs):
        """根据环境实际格式拆分 proprio, sensor, vision"""
        # 需用户实现
        return obs['proprio'], obs['sensor'], obs['vision']

    @torch.no_grad()
    def _compute_ideal_progress(self, z_start, g_target, horizon):
        """使用世界模型和低层策略（确定性）rollout horizon 步，返回总位移"""
        z_cur = z_start.clone()
        for _ in range(horizon):
            dist = self.low_actor(z_cur, g_target)
            a = dist.mean  # 确定性
            z_cur = self.lewm_predictor(z_cur, a)
        return torch.norm(z_cur - z_start, dim=-1)

    @torch.no_grad()
    def _compute_rollout_distance(self, z_start, z_goal, g_target, horizon):
        """使用冻结的世界模型和当前低层策略，rollout horizon 步，返回末端与目标的 L2 距离"""
        z_cur = z_start.clone()
        for _ in range(horizon):
            dist = self.low_actor(z_cur, g_target)   # 返回分布
            a = dist.mean                             # 确定性动作
            z_cur = self.lewm_predictor(z_cur, a)
        return torch.norm(z_cur - z_goal, dim=-1)

    # ------------------- 网络更新函数 -------------------
    def update_low_level(self, batch_size):
        if len(self.buffer.low_buffer) < batch_size:
            return {}
        # 注意：buffer 中的 r 已经包含外部奖励 + 内在奖励
        z, g_target, a, r, z_next, done = self.buffer.sample_low(batch_size)
        z = z.to(self.device)
        g_target = g_target.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        z_next = z_next.to(self.device)
        done = done.to(self.device)

        # ----- 更新 Q 网络（critic 接收隐动作） -----
        td_target = self._calc_low_target(r, z_next, g_target, done)
        q1 = self.low_critic_1(z, g_target, a)
        q2 = self.low_critic_2(z, g_target, a)
        loss_q1 = F.mse_loss(q1, td_target)
        loss_q2 = F.mse_loss(q2, td_target)

        self.low_critic1_opt.zero_grad()
        loss_q1.backward()
        self.low_critic1_opt.step()

        self.low_critic2_opt.zero_grad()
        loss_q2.backward()
        self.low_critic2_opt.step()

        # ----- 更新策略网络 (SAC) -----
        new_dist = self.low_actor(z, g_target)
        new_a = new_dist.rsample()
        log_prob = new_dist.log_prob(new_a).sum(-1)
        q1_new = self.low_critic_1(z, g_target, new_a)
        q2_new = self.low_critic_2(z, g_target, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = -(q_new - self.log_alpha.exp() * log_prob).mean()

        self.low_actor_opt.zero_grad()
        actor_loss.backward()
        self.low_actor_opt.step()

        # 温度参数更新
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()

        # 软更新目标网络
        self._soft_update(self.low_critic_1, self.target_low_critic_1)
        self._soft_update(self.low_critic_2, self.target_low_critic_2)

        return {'low_q_loss': loss_q1.item(), 'low_policy_loss': actor_loss.item()}

    def update_high_level(self, batch_size):
        if len(self.buffer.high_buffer) < batch_size:
            return {}
        z, z_goal, g, sum_r, z_next, dones, _ = self.buffer.sample_high(batch_size)
        z = z.to(self.device)
        z_goal = z_goal.to(self.device)
        g = g.to(self.device)
        sum_r = sum_r.to(self.device)
        z_next = z_next.to(self.device)
        dones = dones.to(self.device)

        # TD3 风格更新
        td_target = self._calc_high_target(sum_r, z_next, g, dones)
        q1 = self.high_critic_1(z, z_goal, g)
        q2 = self.high_critic_2(z, z_goal, g)
        loss_q1 = F.mse_loss(q1, td_target)
        loss_q2 = F.mse_loss(q2, td_target)

        self.high_critic1_opt.zero_grad()
        loss_q1.backward()
        self.high_critic1_opt.step()

        self.high_critic2_opt.zero_grad()
        loss_q2.backward()
        self.high_critic2_opt.step()

        self.update_counter += 1
        if self.update_counter % self.policy_delay == 0:
            new_g = self.high_actor(z, z_goal)
            actor_loss = -self.high_critic_1(z, z_goal, new_g).mean()
            self.high_actor_opt.zero_grad()
            actor_loss.backward()
            self.high_actor_opt.step()
            self._soft_update(self.high_actor, self.target_high_actor)
            self._soft_update(self.high_critic_1, self.target_high_critic_1)
            self._soft_update(self.high_critic_2, self.target_high_critic_2)

        return {'high_q_loss': loss_q1.item()}

    def update_termination_critic(self, batch_size):
        if len(self.buffer.high_buffer) < batch_size:
            return {}
        # 从高层 buffer 采样，因为需要完整的周期数据（z_start, g, z_end）
        z_start, z_goal, g, sum_r, z_end, done, low_trans = self.buffer.sample_high(batch_size)
        # z_start, z_end 是周期首尾隐状态；g 是高层子目标
        z_start = z_start.to(self.device)
        z_end = z_end.to(self.device)
        g_target = self.subgoal_encoder(g.to(self.device))

        # 实际进展
        actual_progress = torch.norm(z_end - z_start, dim=-1)

        # 理想进展：用世界模型 rollout 整个周期（周期长度 = len(low_trans)）
        # 注意周期长度可能变化，需要从 low_trans 中获取
        # 简便方法：假设周期长度为 self.k（固定），或者从 low_trans 长度计算
        # 这里使用固定 k（高层决策频率）
        ideal_progress = self._compute_ideal_progress(z_start, g_target, self.k)

        # 标签：实际进展 < α * 理想进展
        alpha = self.config.term_progress_ratio  # 例如 0.5
        label = (actual_progress < alpha * ideal_progress).float()

        pred = self.termination_critic(z_start, g_target)
        loss = F.binary_cross_entropy(pred, label)
        self.term_opt.zero_grad()
        loss.backward()
        self.term_opt.step()
        return {'term_loss': loss.item()}

    def update_gia_self_supervised(self, batch_size):
        if len(self.buffer.high_buffer) < batch_size:
            return {}
        z, z_goal, g, _, _, _, _ = self.buffer.sample_high(batch_size)
        z = z.to(self.device)
        z_goal = z_goal.to(self.device)
        g_target = self.subgoal_encoder(g.to(self.device))

        with torch.no_grad():
            true_dist = self._compute_rollout_distance(z, z_goal, g_target, self.gia_rollout_horizon)

        uncertainty = torch.zeros_like(true_dist)
        pred_dist = self.gia(z, z_goal, g_target, uncertainty)  # 只返回距离
        loss = F.mse_loss(pred_dist, true_dist)
        self.gia_opt.zero_grad()
        loss.backward()
        self.gia_opt.step()
        self.aux_opt.step()
        return {'gia_self_loss': loss.item()}

    # ------------------- 环境交互 -------------------
    def collect_episode(self, max_steps=1000):
        obs = self.env.reset()
        proprio, sensor, vision = self._parse_obs(obs)
        with torch.no_grad():
            z = self.state_encoder(proprio.unsqueeze(0).to(self.device),
                                   sensor.unsqueeze(0).to(self.device),
                                   vision.unsqueeze(0).to(self.device)).squeeze(0)

        goal_state = torch.from_numpy(self.env.get_goal_state()).float().to(self.device)
        z_goal = self.goal_encoder(goal_state.unsqueeze(0)).squeeze(0)

        g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
        g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)

        low_transitions = []
        stagnation_window = self.config.get('stagnation_window', 10)
        dist_window = deque(maxlen=stagnation_window)
        episode_reward = 0
        step = 0
        done = False

        while not done and step < max_steps:
            z_start = z.clone()
            for _ in range(self.k):
                # 低层动作
                dist = self.low_actor(z.unsqueeze(0), g_target.unsqueeze(0))
                a_prim = dist.sample().squeeze(0)
                a_phys = self.action_decoder(a_prim.unsqueeze(0))
                next_obs, reward, done = self.env.step(a_phys.squeeze(0).cpu().numpy())
                episode_reward += reward
                step += 1

                # 编码下一状态
                proprio, sensor, vision = self._parse_obs(next_obs)
                with torch.no_grad():
                    z_next = self.state_encoder(proprio.unsqueeze(0).to(self.device),
                                                sensor.unsqueeze(0).to(self.device),
                                                vision.unsqueeze(0).to(self.device)).squeeze(0)
                    # 世界模型预测
                    z_pred = self.lewm_predictor(z, a_prim)
                    # 内在奖励：预测误差的负值
                    r_int_wm = - torch.norm(z_next - z_pred, p=2).item()

                # 综合奖励 = 外部奖励 + 权重 * 内在奖励
                r_total = reward + self.lambda_wm * r_int_wm

                # 低层内在奖励（子目标达成度）仅用于记录，不用于 RL 更新
                r_intrinsic_goal = -torch.norm(z_next - (z + g_target), p=2).item()

                # 存储到 buffer（存储综合奖励）
                self.buffer.add_low(z.cpu().numpy(), g_target.cpu().numpy(),
                                    a_prim.cpu().numpy(), r_total,
                                    z_next.cpu().numpy(), done)
                low_transitions.append((z.cpu().numpy(), a_prim.cpu().numpy(),
                                        r_intrinsic_goal, z_next.cpu().numpy()))
                z = z_next

                # 调度器判断终止
                term_prob = self.termination_critic(z.unsqueeze(0), g_target.unsqueeze(0)).item()
                if np.random.random() < term_prob or done:
                    total_r = sum([t[2] for t in low_transitions])   # 子目标内在奖励和
                    self.buffer.add_high(z_start.cpu().numpy(), z_goal.cpu().numpy(),
                                         g.cpu().numpy(), total_r,
                                         z.cpu().numpy(), done, low_transitions)
                    # 重新规划子目标
                    g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                    g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                    low_transitions = []
                    z_start = z.clone()
                    break

                if done:
                    break

            # GIA 更新最终目标（使用预测距离）
            with torch.no_grad():
                uncertainty = torch.zeros(1, device=self.device)
                pred_dist = self.gia(z.unsqueeze(0), z_goal.unsqueeze(0),
                                               g_target.unsqueeze(0), uncertainty)

            dist_window.append(pred_dist)

            if len(dist_window) == stagnation_window:
                half = stagnation_window // 2
                early_avg = np.mean(list(dist_window)[:half])
                recent_avg = np.mean(list(dist_window)[half:])
                if recent_avg >= early_avg:  # 距离未下降，停滞
                    print(f"步骤 {step}: GIA 检测到距离停滞 (早: {early_avg:.3f}, 近: {recent_avg:.3f})，请求新目标")
                    new_goal_state = self.env.request_goal()
                    if isinstance(new_goal_state, np.ndarray):
                        new_goal_state = torch.from_numpy(new_goal_state).float()
                    with torch.no_grad():
                        z_goal = self.goal_encoder(new_goal_state.unsqueeze(0).to(self.device)).squeeze(0)
                    # 目标改变后立即重新规划子目标
                    with torch.no_grad():
                        g = self.high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                        g_target = self.subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                    dist_window.clear()
                low_transitions = []

            if done:
                break

        return episode_reward

    # ------------------- 主训练循环 -------------------
    def train(self, total_episodes):
        for ep in range(total_episodes):
            episode_reward = self.collect_episode()

            # 更新网络
            if len(self.buffer.low_buffer) > self.config.batch_size:
                self.update_low_level(self.config.batch_size)
                self.update_termination_critic(self.config.batch_size)
            if len(self.buffer.high_buffer) > self.config.batch_size:
                if ep % self.config.relabel_freq == 0:
                    self.buffer.relabel_subgoals(self.low_actor, self.k)
                self.update_high_level(self.config.batch_size)

            # 自监督 GIA 训练
            if ep % self.config.gia_train_freq == 0 and len(self.buffer.high_buffer) > self.config.batch_size:
                gia_loss = self.update_gia_self_supervised(self.config.batch_size)
                print(f"Ep {ep}, GIA self-loss: {gia_loss.get('gia_self_loss', 0):.4f}")

            # 保存模型（含 inference.py 需要的全部网络）
            if ep % self.config.save_freq == 0:
                save_dir = Path(self.config.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'state_encoder': self.state_encoder.state_dict(),
                    'goal_encoder': self.goal_encoder.state_dict(),
                    'subgoal_encoder': self.subgoal_encoder.state_dict(),
                    'action_decoder': self.action_decoder.state_dict(),
                    'low_actor': self.low_actor.state_dict(),
                    'high_actor': self.high_actor.state_dict(),
                    'gia': self.gia.state_dict(),
                    'termination_critic': self.termination_critic.state_dict(),
                }, save_dir / f'stage3_ep{ep}.pt')

            print(f"Episode {ep}, Reward: {episode_reward:.2f}")