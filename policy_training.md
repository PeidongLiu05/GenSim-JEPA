# 角色与任务
你是一位精通 PyTorch 和深度强化学习（TD3）的算法工程师。请根据论文公式，实现“源域策略训练（Source Domain Policy Training）”阶段的**损失函数构建模块**。

## 1. 网络结构定义（已知前提）
在计算损失前，假设以下 PyTorch 网络模块已由你实例化并加载参数：
- `F_s`：状态编码器 (State Encoder)，输入 `s_s` (源状态)，输出 `s_z` (潜状态)。
- `G_s`：动作编码器 (Action Encoder)，输入 `s_s` (源状态) 和 `a_s` (源动作)，输出 `a_z` (潜动作)。
- `~F_s`（`F_s_inv`）：状态解码器，输入 `s_z`，输出重建的源状态 `s_s_recon`。
- `~G_s`（`G_s_inv`）：**动作解码器（关键）**，输入 `s_s` (源状态) 和 `a_z` (潜动作)，输出重建的源动作 `a_s_recon`。
- `pi_z`：潜空间策略，输入 `s_z`，输出 `a_z_pred`（用于任务损失）。
- `T_z`：前向潜动力学模型，输入 `(s_z, a_z)`，预测下一潜状态 `s_z_next_pred`。
- `~T_z`（`T_z_inv`）：逆向潜动力学模型，输入 `(s_z, s_z_next)`，预测潜动作 `a_z_inv_pred`。
- `Q_function`：TD3 风格的 Critic 网络（用于计算任务损失梯度）。

## 2. 损失函数构建（核心）
请编写一个函数 `compute_source_losses(batch)`，其中 `batch` 包含从经验回放池采样的张量：`s_s, a_s, r, s_s_next, done`。请严格按照论文公式实现以下三项损失，并返回 `total_loss` 用于反向传播：

### 2.1 任务损失 (Task Loss) —— 公式 (1) & (2)
- 计算潜策略的动作：`a_z_pred = pi_z(F_s(s_s))`。
- 通过动作解码器还原源域动作：`a_s_pred = ~G_s(s_s, a_z_pred)`（**注意：这里必须传入源状态 `s_s` 和潜动作 `a_z_pred`**）。
- 使用 Critic 网络评估 `(s_s, a_s_pred)` 的 Q 值。
- 策略梯度（Policy Gradient）的损失定义为负 Q 值（TD3 标准做法）：`loss_task = - Q_function(s_s, a_s_pred).mean()`。
- **梯度流要求**：此损失必须能够反向传播，通过 `~G_s` 更新动作解码器，通过 `pi_z` 更新潜策略，并通过 `F_s` 更新状态编码器。

### 2.2 潜动态损失 (Latent Dynamics Loss) —— 公式 (3)
- 编码当前状态和动作：`s_z = F_s(s_s)`，`a_z = G_s(s_s, a_s)`。
- 编码下一时刻状态：`s_z_next = F_s(s_s_next)`。
- 前向预测：`s_z_next_pred = T_z(s_z, a_z)`。计算损失 `loss_forward = MSE(s_z_next_pred, s_z_next)`。
- 逆向预测：`a_z_inv_pred = ~T_z(s_z, s_z_next)`。计算损失 `loss_backward = MSE(a_z_inv_pred, a_z)`。
- 最终动态损失：`loss_dyn = loss_forward + loss_backward`。

### 2.3 重建损失 (Reconstruction Loss) —— 公式 (4)
- 状态重建：`s_s_recon = ~F_s(F_s(s_s))`。计算 `loss_state_recon = MSE(s_s_recon, s_s)`。
- 动作重建（**关键细节**）：首先通过动作编码器得到 `a_z = G_s(s_s, a_s)`，再通过动作解码器重建：`a_s_recon = ~G_s(s_s, a_z)`。计算 `loss_action_recon = MSE(a_s_recon, a_s)`。
- 最终重建损失：`loss_rec = loss_state_recon + loss_action_recon`。

## 3. 总损失聚合与优化策略
- 虽然论文中提到可以交替更新，但为简化实现，请将三个损失直接加权求和：`total_loss = loss_task + loss_dyn + loss_rec`。
- 在 `total_loss.backward()` 时，确保所有涉及的网络（`F_s`, `G_s`, `~F_s`, `~G_s`, `pi_z`, `T_z`, `~T_z`）的 `requires_grad` 均为 True，并接收梯度。注意：Critic 网络（`Q_function`）在策略损失更新时不计入此总损失的反向传播（通常 Critic 由单独的 TD3 更新步骤优化，但公式 (2) 指明梯度来自 Q 对动作的导数，请确保 `loss_task` 中调用 `Q_function` 时使用 `.detach()` 或合理处理计算图，以仅更新 Actor 侧参数）。

## 4. 额外注意事项（防止代码Bug）
- **动作解码器输入**：`~G_s` 是一个**条件解码器**，它必须同时接收源状态 `s_s` 和潜动作 `a_z`。请勿将其误认为仅接收潜动作。
- **动作编码器输入**：`G_s` 接收的是 `(s_s, a_s)`，而不是 `(s_z, a_s)`（因为论文公式 (4) 明确写为 `G^s(s_k^s, a_k^s)`）。
- 所有损失均使用 **均方误差 (MSE Loss)** 或 **L1 Loss**，请严格遵循公式中使用的范数（公式 3/4 使用 L2，公式 9 后续才用 L1）。
- 请注明 `s_s` 的状态维度包含机器人关节（正余弦）、夹爪宽度、物体位置等；`a_s` 为关节速度控制指令。