import torch
import torch.nn as nn

class TransformerPredictor(nn.Module):

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim = None,
        dim_head = 64,
        dropout = 0.0,
        emb_dropout = 0.0
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock
        )

    def forward(self, x, c):
        T = x.size()
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x