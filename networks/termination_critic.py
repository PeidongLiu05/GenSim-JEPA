import torch
import torch.nn as nn

class TerminationCritic(nn.Module):
    def __init__(self, state_dim=128, subgoal_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + subgoal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, z, g_target):
        x = torch.cat([z, g_target], dim=-1)
        logit = self.net(x)  # 原始输出 (..., 1)
        return torch.sigmoid(logit).squeeze(-1)