import torch
import torch.nn as nn

class GoalInfoAcquirer(nn.Module):
    def __init__(self, state_dim=128, goal_dim=128, subgoal_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim + subgoal_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)   # 输出预测距离
        )
    def forward(self, z, z_goal, g_target, uncertainty):
        x = torch.cat([z, z_goal, g_target, uncertainty.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)   # 返回距离