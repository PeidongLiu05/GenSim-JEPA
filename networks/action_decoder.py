import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    将低层策略输出的原始动作（低维）解码为物理控制信号和持续步数
    """
    def __init__(self, prim_dim=10, hidden_dim=64, action_dim=8, duration_dim=1):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(prim_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim + duration_dim)
        )
    def forward(self, a_prim):
        a_phys = self.net(a_prim)
        action = a_phys[..., : self.action_dim]
        duration = torch.sigmoid(a_phys[..., self.action_dim:])  # 输出(0,1)然后映射到步数范围
        return action, duration