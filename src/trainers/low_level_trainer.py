import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class LowLevelTrainer:
    def __init__(self, policy, q1, q2, target_q1, target_q2, 
                 action_decoder, gamma=0.99, tau=0.005, lr=3e-4):
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.action_decoder = action_decoder
        self.gamma = gamma
        self.tau = tau
        self.policy_optim = optim.Adam(policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(q2.parameters(), lr=lr)
        
    def update(self, replay_buffer, batch_size=256):
        z, g, a_prim, r, z_next, done = replay_buffer.sample_low(batch_size)
        
        # 计算目标Q值
        with torch.no_grad():
            next_dist = self.policy(z_next, g)
            next_a = next_dist.rsample()
            next_q1 = self.target_q1(z_next, g, next_a)
            next_q2 = self.target_q2(z_next, g, next_a)
            next_q = torch.min(next_q1, next_q2)
            target_q = r + self.gamma * (1 - done) * next_q
        
        # 更新Q网络
        q1_pred = self.q1(z, g, a_prim)
        q2_pred = self.q2(z, g, a_prim)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2_pred, target_q)
        
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        
        # 更新策略
        new_dist = self.policy(z, g)
        new_a = new_dist.rsample()
        new_q1 = self.q1(z, g, new_a)
        new_q2 = self.q2(z, g, new_a)
        new_q = torch.min(new_q1, new_q2)
        policy_loss = (new_dist.entropy().mean() * 0.1 - new_q.mean())
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # 软更新目标网络
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        
        return {"q_loss": q1_loss.item(), "policy_loss": policy_loss.item()}
    
    def _soft_update(self, source, target):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)