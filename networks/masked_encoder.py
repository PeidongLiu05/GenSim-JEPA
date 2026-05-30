import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from networks.utils.sigreg import (
    get_sincos_pos_embed,
    interpolate_pos_embed,
)
from networks.utils.sigreg import PatchMasking


class MaskedEncoderOutput:
    encoded: torch.Tensor
    mask: torch.Tensor
    ids_keep: torch.Tensor
    grid_size: Tuple[int, int]


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP层"""

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """随机深度路径"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """图像到patch的嵌入"""

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        if isinstance(img_size, int):
            self.grid_size = (img_size // patch_size, img_size // patch_size)
        else:
            self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedEncoder(nn.Module):

    def __init__(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        masking: Optional[PatchMasking] = None,
        dynamics_img_size: bool = False,
        norm_layer: Optional[nn.Module] = None,
        use_cls_token: bool = True,
        use_learnable_pos_embed: bool = False
    ):
        super().__init__()

        self.dynamics_img_size = dynamics_img_size
        self.masking = masking
        self.use_cls_token = use_cls_token
        self.use_learnable_pos_embed = use_learnable_pos_embed

        norm_layer = norm_layer or nn.LayerNorm

        # Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # 位置嵌入
        num_patches = self.patch_embed.num_patches
        if use_learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if use_cls_token else 0), embed_dim))
        else:
            # 使用固定的正弦余弦位置嵌入
            pos_embed = get_sincos_pos_embed(
                embed_dim, num_patches, mode="2d",
                grid_size=self.patch_embed.grid_size,
                cls_token=use_cls_token
            )
            self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0))

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_prefix_tokens = 1
        else:
            self.cls_token = None
            self.num_prefix_tokens = 0

        # Transformer块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
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

        # 最终归一化
        self.norm = norm_layer(embed_dim)

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 缓存属性
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        self.patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        self.default_grid_h, self.default_grid_w = self.patch_embed.grid_size
        self.has_class_token = use_cls_token
        self.num_reg_tokens = 0
        self.no_embed_class = False

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.use_learnable_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 初始化patch嵌入
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 初始化线性层
        self.apply(self._init_weights_hook)

    def _init_weights_hook(self, m):
        """权重初始化hook"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_grid_size(self, images: torch.Tensor) -> Tuple[int, int]:
        """从图像维度计算patch网格大小"""
        H, W = images.shape[-2:]
        return H // self.patch_size_h, W // self.patch_size_w

    def _get_pos_embed(
            self, grid_h: int, grid_w: int
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """获取位置嵌入，支持动态大小的插值"""
        pos_embed = self.pos_embed
        num_prefix = self.num_prefix_tokens if not self.no_embed_class else 0

        if self.dynamic_img_size and (
                grid_h != self.default_grid_h or grid_w != self.default_grid_w
        ):
            src_patches = pos_embed.shape[1] - num_prefix
            src_size = int(src_patches ** 0.5)
            pos_embed = interpolate_pos_embed(
                pos_embed, (src_size, src_size), (grid_h, grid_w), num_prefix
            )

        if self.no_embed_class:
            return None, pos_embed
        return (
            pos_embed[:, : self.num_prefix_tokens],
            pos_embed[:, self.num_prefix_tokens:],
        )

    def _get_prefix_tokens(self, B: int) -> Optional[torch.Tensor]:
        """获取CLS token，扩展到batch大小"""
        if self.has_class_token and self.cls_token is not None:
            return self.cls_token.expand(B, -1, -1)
        return None

    def forward(self, images: torch.Tensor) -> MaskedEncoderOutput:
        """编码图像，支持可选的掩码

        Args:
            images: 输入图像 (B, C, H, W)

        Returns:
            MaskedEncoderOutput: 包含编码token和掩码信息
        """
        B = images.shape[0]
        device = images.device

        # 计算网格大小
        grid_h, grid_w = self._get_grid_size(images)
        num_patches = grid_h * grid_w

        # Patch嵌入 + 位置嵌入
        x = self.patch_embed(images)
        prefix_pos, patch_pos = self._get_pos_embed(grid_h, grid_w)
        x = x + patch_pos

        # 应用掩码（仅训练时）
        if self.training and self.masking is not None:
            mask_out = self.masking(x, grid_h, grid_w)
            x = mask_out.visible
            mask = mask_out.mask
            ids_keep = mask_out.ids_keep
        else:
            mask = torch.zeros(B, num_patches, device=device)
            ids_keep = (
                torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)
            )

        # 添加CLS token
        prefix = self._get_prefix_tokens(B)
        if prefix is not None:
            if prefix_pos is not None and not self.no_embed_class:
                prefix = prefix + prefix_pos
            x = torch.cat([prefix, x], dim=1)

        # Transformer块处理
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return MaskedEncoderOutput(
            encoded=x,
            mask=mask,
            ids_keep=ids_keep,
            grid_size=(grid_h, grid_w),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """无掩码编码（用于推理）"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
        if was_training:
            self.train()
        return output.encoded

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"patch_size=({self.patch_size_h}, {self.patch_size_w}), "
            f"depth={len(self.blocks)}, "
            f"num_heads={self.blocks[0].attn.num_heads}, "
            f"num_prefix_tokens={self.num_prefix_tokens}, "
            f"has_masking={self.masking is not None}"
        )
