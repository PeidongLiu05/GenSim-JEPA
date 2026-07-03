import torch
import torch.nn as nn

class LeWMPredictor(nn.Module):
    def __init__(self, latent_dim=128, action_dim=29, hidden=512, num_layers=4,
                 num_heads=8, embed_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + action_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden,
                                       batch_first=True),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, z_t, a_t):
        # z_t: (B, latent_dim), a_t: (B, action_dim)
        x = torch.cat([z_t, a_t], dim=-1)          # (B, latent_dim + action_dim)
        x = self.input_proj(x)                     # (B, embed_dim)
        x = x.unsqueeze(1)                         # (B, 1, embed_dim)
        x = self.transformer(x)                    # (B, 1, embed_dim)
        x = x.squeeze(1)                           # (B, embed_dim)
        return self.output_proj(x)                 # (B, latent_dim)