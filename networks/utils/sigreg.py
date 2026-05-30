import torch

class SIGReg(torch.nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (SIGReg)
    用于强制隐状态分布接近各向同性高斯分布 N(0, I)
    
    输入形状: (B, D) 其中 B=batch_size, D=latent_dim
    输出: 标量正则化损失
    """
    def __init__(self, knots=17, num_proj=1024):
        """
        knots: 用于数值积分的节点数
        num_proj: 随机投影的数量
        """
        super().__init__()
        self.num_proj = num_proj
        
        # 定义积分区间 [0, 3] 上的节点 t 和权重 weights
        # t: (knots,)  从 0 到 3 的等间距点
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)                     # 积分步长
        # 梯形法则权重：两端为 dt/2，中间为 dt
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # 标准高斯窗口函数 exp(-t^2/2)  (knots,)
        window = torch.exp(-t.square() / 2.0)
        
        # 注册为 buffer，不参与梯度更新
        self.register_buffer("t", t)                # (knots,)
        self.register_buffer("phi", window)         # (knots,)
        self.register_buffer("weights", weights * window)  # (knots,)

    def forward(self, proj):
        """
        输入 proj: (B, D)  其中 B = batch_size, D = latent_dim
        输出: 标量正则化损失
        """
        # 添加时间维度，将其视为 T=1 的时间序列，便于统一计算
        # 原 proj: (B, D) -> 添加维度 -> (T, B, D)  T=1
        proj = proj.unsqueeze(0)        # (1, B, D)
        T, B, D = proj.shape            # T=1, B=batch_size, D=latent_dim

        # 1. 生成随机投影矩阵 A (D, num_proj)，每一列归一化为单位向量
        #    A 用于将高维隐状态投影到低维（num_proj 维）空间
        A = torch.randn(D, self.num_proj, device=proj.device)   # (D, num_proj)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True))             # (D, num_proj)，每列模为1

        # 2. 计算投影值 x_t：原始隐状态 @ A，然后乘以时间节点 t
        #    proj @ A: (T, B, D) @ (D, num_proj) -> (T, B, num_proj)
        #    然后 unsqueeze(-1): (T, B, num_proj, 1)
        #    再乘以 self.t: (knots,) -> 广播为 (T, B, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t   # (T, B, num_proj, knots)

        # 3. 计算 Epps-Pulley 统计量
        #    cos(x_t) 和 sin(x_t): 形状 (T, B, num_proj, knots)
        #    在时间维度 T 上求平均（这里 T=1，但为了通用性保留），得到 (B, num_proj, knots)
        cos_mean = x_t.cos().mean(dim=0)          # (B, num_proj, knots)
        sin_mean = x_t.sin().mean(dim=0)          # (B, num_proj, knots)

        # 4. 计算与标准高斯窗口的误差平方和
        #    cos_mean - self.phi: (B, num_proj, knots) 减去 (knots,) -> 广播
        err = (cos_mean - self.phi).square() + sin_mean.square()   # (B, num_proj, knots)

        # 5. 加权积分（沿 knots 维度）
        #    err @ self.weights: (B, num_proj, knots) 点乘 (knots,) -> (B, num_proj)
        statistic = (err @ self.weights) * B      # 乘以 B 相当于原论文中的 proj.size(-2)

        # 6. 对所有投影和批次求平均（标量）
        return statistic.mean()                  # 标量