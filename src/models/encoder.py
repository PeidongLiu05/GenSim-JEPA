import torch.nn as nn


class ActionsEncoder(nn.Module):

    def __init__(
            self,
            input_dim = 10,
            smoothed_dim = 10,
            emb_dim = 10,
            mlp_scale = 4
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim)
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x