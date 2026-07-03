import torch
import torch.nn as nn
from .utils.weight_init import weight_init

class ActionDecoder(nn.Module):
    def __init__(self, proprio_dim=68, prim_dim=10, hidden_dim=64, action_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim + prim_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, proprio, a_prim):
        return self.net(torch.cat([proprio, a_prim], dim=-1))
