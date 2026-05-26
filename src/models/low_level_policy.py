import torch
import torch.nn as nn
from torch.distributions import Normal

class LowLevelPolicy(nn.Module):
    def __init__(self, z_dim=128, g_dim=128, prim_dim=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + g_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden, prim_dim)
        self.log_std = nn.Parameter(torch.zeros(prim_dim))
        
    def forward(self, z, g_target, deterministic=False):
        x = torch.cat([z, g_target], dim=-1)
        h = self.net(x)
        mean = self.mean(h)
        if deterministic:
            return mean, None
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist