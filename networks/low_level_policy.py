import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .utils.weight_init import weight_init

class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim=128, goal_dim=128, hidden_dim=256, action_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, z, g_target, deterministic=False):
        x = torch.cat([z, g_target], dim=-1)
        h = self.net(x)
        mean = self.mean(h)
        if deterministic:
            return mean, None

        std = F.softplus(self.log_std(h))
        dist = Normal(mean, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample).sum(dim=-1, keepdim=True)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        return action, log_prob
