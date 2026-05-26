import torch.nn as nn

class SubgoalEncoder(nn.Module):
    def __init__(self, g_dim=128, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim)
        )
    def forward(self, g):
        return self.net(g)