"""
Vision Transformer (ViT) 编码器，无 CLS token，输出所有 patch tokens 的均值。
使用固定的正弦余弦2D位置编码，支持动态图像尺寸的插值。
"""

import torch
import torch.nn as nn
from typing import Optional

from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """将图像切分为 patches 并线性投影到嵌入维度"""
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch, W//patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """多头自注意力"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    ViT 编码器，无 CLS token，输出所有 patch tokens 的均值。
    使用固定的正弦余弦2D位置编码（可插值），适合低分辨率图像。
    """
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Patch 嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # 固定的正弦余弦2D位置编码（无 CLS token）
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token=False)  # (H*W, D)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D)

        # 随机深度 dropout 率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer 编码器层
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # 初始化 patch embedding 的卷积
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        # 初始化 Transformer 层
        for block in self.blocks:
            nn.init.trunc_normal_(block.attn.qkv.weight, std=0.02)
            nn.init.constant_(block.attn.qkv.bias, 0)
            nn.init.trunc_normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.trunc_normal_(block.mlp.fc2.weight, std=0.02)

    def interpolate_pos_encoding(self, new_img_size: int):
        """
        当输入图像尺寸变化时，插值位置编码。
        需在 forward 前调用（可选，若图像尺寸与训练时一致则无需调用）。
        """
        new_grid_h = new_img_size // self.patch_size
        new_grid_w = new_img_size // self.patch_size
        if new_grid_h == self.grid_size[0] and new_grid_w == self.grid_size[1]:
            return
        old_pos = self.pos_embed[0]  # (N, D)
        # 重塑为 2D 网格
        old_pos = old_pos.reshape(1, self.grid_size[0], self.grid_size[1], -1)
        # 插值
        new_pos = nn.functional.interpolate(
            old_pos.permute(0, 3, 1, 2),
            size=(new_grid_h, new_grid_w),
            mode='bicubic',
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(new_grid_h * new_grid_w, -1)
        self.register_buffer("pos_embed", new_pos.unsqueeze(0))
        self.grid_size = (new_grid_h, new_grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入图像 (B, C, H, W)
        :return: 图像特征 (B, embed_dim) —— 所有 patch tokens 的均值
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # 添加位置编码
        x = x + self.pos_embed[:, :x.shape[1], :]

        # 经过 Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 全局平均池化（等价于 mean pooling over patches）
        x = x.mean(dim=1)  # (B, D)
        return x