class TransitionPredictor(nn.Module):
    def __init__(self, z_dim=128, a_prim_dim=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_prim_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
    def forward(self, z, a_prim):
        return self.net(torch.cat([z, a_prim], dim=-1))