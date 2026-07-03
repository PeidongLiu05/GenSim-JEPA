import torch
import torch.nn as nn
from .utils.weight_init import weight_init

class ActionEncoder(nn.Module):
    def __init__(self, proprio_dim=68, action_dim=29, hidden_dim=64, prim_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, prim_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, proprio, action):
        return self.net(torch.cat([proprio, action], dim=-1))
