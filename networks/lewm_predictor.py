import torch
import torch.nn as nn

class LeWMPredictor(nn.Module):
    def __init__(self, latent_dim=128, action_dim=10, hidden=512, num_layers=4, num_heads=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim + action_dim, nhead=num_heads, dim_feedforward=hidden),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(latent_dim + action_dim, latent_dim)

    def forward(self, z_t, a_t):
        # z_t: (B, latent_dim), a_t: (B, action_dim)
        x = torch.cat([z_t, a_t], dim=-1).unsqueeze(0)  # (1, B, D)
        x = self.transformer(x).squeeze(0)              # (B, D)
        return self.output_proj(x)