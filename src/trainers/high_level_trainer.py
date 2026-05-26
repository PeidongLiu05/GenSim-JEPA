class HighLevelTrainer:
    def __init__(self, high_policy, high_q, target_high_q, 
                 low_policy, subgoal_encoder, 
                 gamma=0.99, tau=0.005, lr=3e-4, k=10):
        self.high_policy = high_policy
        self.high_q = high_q
        self.target_high_q = target_high_q
        self.low_policy = low_policy
        self.subgoal_encoder = subgoal_encoder
        self.gamma = gamma
        self.tau = tau
        self.k = k  # 子目标步数
        self.policy_optim = optim.Adam(high_policy.parameters(), lr=lr)
        self.q_optim = optim.Adam(high_q.parameters(), lr=lr)
        
    def update(self, replay_buffer, batch_size=128):
        # 获取高层经验（已经过relabeling）
        z, z_goal, g, z_next, r, done, low_trans = replay_buffer.sample_high(batch_size)
        
        # 将g映射为g_target（用于计算Q值中的动作）
        g_target = self.subgoal_encoder(g)
        
        # 目标Q值
        with torch.no_grad():
            next_g = self.high_policy(z_next, z_goal)
            next_g_target = self.subgoal_encoder(next_g)
            next_q = self.target_high_q(z_next, z_goal, next_g_target)
            target_q = r + self.gamma * (1 - done) * next_q
        
        # 更新Q网络
        q_pred = self.high_q(z, z_goal, g_target)
        q_loss = torch.nn.functional.mse_loss(q_pred, target_q)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()
        
        # 更新高层策略
        new_g = self.high_policy(z, z_goal)
        new_g_target = self.subgoal_encoder(new_g)
        new_q = self.high_q(z, z_goal, new_g_target)
        policy_loss = -new_q.mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # 软更新目标Q
        for p, tp in zip(self.high_q.parameters(), self.target_high_q.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        return {"high_q_loss": q_loss.item(), "high_policy_loss": policy_loss.item()}