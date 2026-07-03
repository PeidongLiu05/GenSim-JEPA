import torch
import torch.nn as nn
from .utils.vision_transformer import VisionTransformer
from .utils.weight_init import weight_init


class StateEncoder(nn.Module):
    def __init__(self, 
                 proprio_dim=72,
                 sensor_dim=38,
                 vision_c=3, vision_h=64, vision_w=64,
                 embed_dim=128,
                 depth=4, num_heads=4):
        super().__init__()
        self.vision_encoder = VisionTransformer(
            img_size=vision_h,
            patch_size=8,
            in_chans=vision_c,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, embed_dim),
            nn.Tanh(),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, embed_dim),
            nn.Tanh(),
        )
        self.fusion = nn.Linear(embed_dim * 3, embed_dim)
        self.apply(weight_init)

    def forward(self, proprio, sensor, vision):
        z_prop = self.proprio_encoder(proprio)
        z_sens = self.sensor_encoder(sensor)
        z_vis = self.vision_encoder(vision)
        z = torch.cat([z_prop, z_sens, z_vis], dim=-1)
        z = self.fusion(z)
        return z
