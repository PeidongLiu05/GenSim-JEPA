import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    将低层策略输出的原始动作（低维）解码为物理控制信号和持续步数
    """
    def __init__(self, prim_dim=10, action_dim=8, duration_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prim_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim + duration_dim)
        )
    def forward(self, a_prim):
        out = self.net(a_prim)
        action = out[..., :action_dim]
        duration = torch.sigmoid(out[..., action_dim:])  # 输出(0,1)然后映射到步数范围
        return action, duration