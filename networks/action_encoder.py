import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self, action_dim=8, duration_dim=1, hidden_dim=64, prim_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + duration_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prim_dim)
        )

    def forward(self, action, duration):
        a_prim = self.net(torch.cat([action, duration], dim=-1))
        return a_prim