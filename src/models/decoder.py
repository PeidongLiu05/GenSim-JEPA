import torch.nn as nn

class ActionsEncoder(nn.Module):

    def __init__(
        self,
        emb_dim=10,
        smoothed_dim = 10,
        output_dim=10,
        mlp_scale = 4
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(emb_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, smoothed_dim)
        )
        self.patch_embed = nn.Conv1d(smoothed_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = x.float()
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x