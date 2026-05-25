from dataclasses import dataclass
import math
import timm
from timm import DropPath, Mlp, trunc_normal_
import torch.nn as nn
from typing import Optional, Tuple, Union
from .utils.patch_masking import PatchMasking


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
        maksing: Optional[PatchMasking] = None,
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

        self.patch_embed = PatchEmbed()