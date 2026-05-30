import torch.nn as nn

class GoalEncoder(nn.Module):
    """
    将任务终点状态（例如目标位置、物体姿态）编码为z_goal
    终点状态用与状态相同的表示形式（但可以是理想状态）
    """
    def __init__(self, state_dim=128, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, goal_state):
        return self.net(goal_state)