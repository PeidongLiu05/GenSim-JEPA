import torch
import torch.nn as nn
from .utils.VisionTransformer import VisionTransformer


class StateEncoder(nn.Module):
    """
    将多模态观测（本体感觉、传感器、低分辨率视觉）编码为隐状态z_t
    使用MaskedEncoder作为视觉部分的编码器，并结合其他传感器特征
    """
    def __init__(self, 
                 proprio_dim=10,    # 位移、速度、加速度、角动量、关节状态
                 sensor_dim=5,      # 陀螺仪与雷达信号
                 vision_c=3, vision_h=64, vision_w=64,
                 embed_dim=128,
                 depth=4, num_heads=4):
        super().__init__()
        # 视觉编码器（你的MaskedEncoder，训练时禁用masking）
        self.vision_encoder = VisionTransformer(
            img_size=(vision_h, vision_w),
            patch_size=8,
            in_chans=vision_c,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            masking=None  # 推理和RL训练时不用mask
        )
        # 本体系数编码器
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        # 传感器编码器
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        # 融合层
        self.fusion = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, proprio, sensor, vision):
        # proprio: (B, proprio_dim)
        # sensor: (B, sensor_dim)
        # vision: (B, C, H, W)
        z_prop = self.proprio_encoder(proprio)
        z_sens = self.sensor_encoder(sensor)
        # vision编码器输出patch tokens，取mean作为全局表示
        vision_tokens = self.vision_encoder(vision)  # shape: (B, N, D)
        z_vis = vision_tokens.mean(dim=1)
        # 拼接并融合
        z = torch.cat([z_prop, z_sens, z_vis], dim=-1)
        z = self.fusion(z)
        return z