import torch.nn as nn

class TerminationCritic(nn.Module):
    def __init__(self, z_dim=128, g_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + g_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, z, g_target):
        x = torch.cat([z, g_target], dim=-1)
        return self.net(x).squeeze(-1)