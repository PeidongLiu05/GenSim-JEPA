import torch.nn as nn

class HighLevelPolicy(nn.Module):
    def __init__(self, z_dim=128, goal_dim=128, g_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, g_dim)
        )

    def forward(self, z, z_goal):
        x = torch.cat([z, z_goal], dim=-1)
        g = self.net(x)
        return g