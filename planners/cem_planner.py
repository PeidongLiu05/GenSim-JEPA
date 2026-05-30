import torch
import torch.nn as nn
import numpy as np

class CEMPlanner(nn.Module):
    """
    CEM 规划器，用于在隐空间中进行动作序列规划。
    
    给定当前隐状态 z_t 和目标隐状态 z_goal，CEM 通过迭代优化采样分布，
    找出一段动作序列，使得预测出的未来状态最接近目标。
    
    Args:
        predictor: LeWorldModel 预测器，接受 (z, a) 返回 z_next。
        action_dim: 动作空间的维度。
        horizon: 规划的时间步长（动作序列长度）。
        num_samples: 每次迭代采样的动作序列数量。
        num_elites: 保留的最优序列数量（用于更新分布）。
        iterations: CEM 迭代次数。
        action_low: 每个动作维度的下限（可以是标量或数组）。
        action_high: 每个动作维度的上限。
        device: 计算设备。
    """
    def __init__(self,
                 predictor: nn.Module,
                 action_dim: int,
                 horizon: int = 12,
                 num_samples: int = 300,
                 num_elites: int = 30,
                 iterations: int = 5,
                 action_low: float = -1.0,
                 action_high: float = 1.0,
                 device: str = "cuda"):
        super().__init__()
        self.predictor = predictor
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.device = device

        # 动作边界
        if isinstance(action_low, (int, float)):
            self.action_low = torch.full((action_dim,), action_low, dtype=torch.float32, device=device)
        else:
            self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        if isinstance(action_high, (int, float)):
            self.action_high = torch.full((action_dim,), action_high, dtype=torch.float32, device=device)
        else:
            self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # 初始分布参数（均值为0，标准差0.5）
        self.register_buffer("init_mean", torch.zeros(horizon, action_dim, device=device))
        self.register_buffer("init_std", torch.full((horizon, action_dim), 0.5, device=device))

    @torch.no_grad()
    def plan(self, z_current: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        规划动作序列，返回第一步动作（用于 MPC）。

        Args:
            z_current: 当前隐状态，shape (batch, latent_dim)
            z_goal: 目标隐状态，shape (batch, latent_dim)

        Returns:
            a_prim: 最佳动作序列的第一步，shape (batch, action_dim)
        """
        # 确保输入为2D，并放在正确设备上
        z_current = z_current.to(self.device)
        z_goal = z_goal.to(self.device)
        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)

        batch_size = z_current.shape[0]

        # 初始化动作分布
        mean = self.init_mean.unsqueeze(0).expand(batch_size, -1, -1).clone()
        std = self.init_std.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for _ in range(self.iterations):
            # 1. 从当前分布采样动作序列
            # 采样 shape: (batch, num_samples, horizon, action_dim)
            samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                batch_size, self.num_samples, self.horizon, self.action_dim, device=self.device
            )
            # 裁剪到动作边界
            samples = torch.clamp(samples, self.action_low, self.action_high)

            # 2. 计算每条轨迹的代价
            costs = torch.zeros(batch_size, self.num_samples, device=self.device)
            for i in range(self.num_samples):
                # 逐个样本 rollout（并行化受限，但 horizon 不长时可接受）
                z = z_current.clone()
                for t in range(self.horizon):
                    a_t = samples[:, i, t, :]           # (batch, action_dim)
                    z = self.predictor(z, a_t)          # (batch, latent_dim)
                # 计算轨迹末端与目标的欧氏距离作为代价
                dist = torch.norm(z - z_goal, dim=-1)   # (batch,)
                costs[:, i] = dist

            # 3. 选择精英样本
            elite_indices = torch.topk(costs, self.num_elites, dim=1, largest=False).indices  # (batch, num_elites)

            # 4. 根据精英样本更新分布
            new_mean = torch.zeros_like(mean)
            new_std = torch.zeros_like(std)
            for b in range(batch_size):
                elites = samples[b, elite_indices[b], :, :]  # (num_elites, horizon, action_dim)
                new_mean[b] = elites.mean(dim=0)
                new_std[b] = elites.std(dim=0) + 1e-6       # 避免零标准差

            # 平滑更新（可选）
            mean = 0.9 * mean + 0.1 * new_mean
            std = 0.9 * std + 0.1 * new_std

        # 最终选择最佳动作序列（精英中代价最小的第一个序列）
        # 使用最后一次迭代的采样结果，取第一个精英序列的第一步
        with torch.no_grad():
            # 重新采样一次最佳分布
            best_samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                batch_size, 1, self.horizon, self.action_dim, device=self.device
            )
            best_samples = torch.clamp(best_samples, self.action_low, self.action_high)
            # 计算代价
            best_costs = torch.zeros(batch_size, 1, device=self.device)
            z = z_current.clone()
            for t in range(self.horizon):
                a_t = best_samples[:, 0, t, :]
                z = self.predictor(z, a_t)
            best_costs[:, 0] = torch.norm(z - z_goal, dim=-1)
            best_idx = torch.argmin(best_costs, dim=1)  # 理论上总是0，但保留结构

        # 返回第一步动作
        return best_samples[torch.arange(batch_size), 0, 0, :]  # (batch, action_dim)


# 可选：更高效的批处理版本（使用 torch.vmap 或多次并行，但保持简单）
class CEMPlannerFast(CEMPlanner):
    """
    将 rollouts 并行化，利用 torch 的批处理能力一次性 rollout 所有样本。
    适用于 horizon 较小且 GPU 内存充足的情况。
    """
    @torch.no_grad()
    def plan(self, z_current: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        # 与父类类似，但将循环内对 num_samples 的迭代改为全批量处理
        z_current = z_current.to(self.device)
        z_goal = z_goal.to(self.device)
        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)

        batch_size = z_current.shape[0]
        mean = self.init_mean.unsqueeze(0).expand(batch_size, -1, -1).clone()
        std = self.init_std.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for _ in range(self.iterations):
            # 采样 shape: (batch, num_samples, horizon, action_dim)
            samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                batch_size, self.num_samples, self.horizon, self.action_dim, device=self.device
            )
            samples = torch.clamp(samples, self.action_low, self.action_high)

            # 并行 rollout: 将 batch 和 num_samples 合并为一个大 batch
            # shape: (batch * num_samples, horizon, action_dim)
            flat_samples = samples.view(batch_size * self.num_samples, self.horizon, self.action_dim)
            flat_z = z_current.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(batch_size * self.num_samples, -1)
            flat_z_goal = z_goal.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(batch_size * self.num_samples, -1)

            z = flat_z
            for t in range(self.horizon):
                a_t = flat_samples[:, t, :]
                z = self.predictor(z, a_t)
            dist = torch.norm(z - flat_z_goal, dim=-1)  # (batch * num_samples,)
            costs = dist.view(batch_size, self.num_samples)  # (batch, num_samples)

            # 选择精英
            elite_indices = torch.topk(costs, self.num_elites, dim=1, largest=False).indices
            new_mean = torch.zeros_like(mean)
            new_std = torch.zeros_like(std)
            for b in range(batch_size):
                elites = samples[b, elite_indices[b], :, :]  # (num_elites, horizon, action_dim)
                new_mean[b] = elites.mean(dim=0)
                new_std[b] = elites.std(dim=0) + 1e-6
            mean = 0.9 * mean + 0.1 * new_mean
            std = 0.9 * std + 0.1 * new_std

        # 最终选择（同样采用并行计算）
        best_samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
            batch_size, 1, self.horizon, self.action_dim, device=self.device
        )
        best_samples = torch.clamp(best_samples, self.action_low, self.action_high)
        flat_best = best_samples.view(batch_size, self.horizon, self.action_dim)
        z = z_current.clone()
        for t in range(self.horizon):
            z = self.predictor(z, flat_best[:, t, :])
        best_costs = torch.norm(z - z_goal, dim=-1)  # (batch,)
        best_idx = torch.argmin(best_costs)  # 实际只有一个候选，但保留
        return flat_best[torch.arange(batch_size), 0, :]  # 第一步动作