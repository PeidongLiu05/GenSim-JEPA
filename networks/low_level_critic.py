import torch
import torch.nn as nn

class LowLevelCritic(nn.Module):
    def __init__(self, state_dim=128, goal_dim=128, hidden_dim=256, action_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z, g_target, a):
        x = torch.cat([z, g_target, a], dim=-1)
        return self.net(x)