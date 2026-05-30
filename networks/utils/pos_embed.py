"""Positional embedding utilities for vision transformers."""

import torch


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings for image patches.

    :param embed_dim: Embedding dimension (must be divisible by 4)
    :param grid_size: Grid height/width as int (square) or (height, width) tuple
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings of shape (H*W, embed_dim) or
             (H*W + 1, embed_dim) if cls_token=True
    """
    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be positive and divisible by 4, got {embed_dim}"
        )

    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid dimensions must be positive, got ({grid_h}, {grid_w})")

    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)

    dim = embed_dim // 4
    omega = torch.arange(dim, dtype=torch.float32) / dim
    omega = 1.0 / (10000**omega)

    out_h = grid[:, 0:1] @ omega.unsqueeze(0)
    out_w = grid[:, 1:2] @ omega.unsqueeze(0)

    pe = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)],
        dim=1,
    )

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe