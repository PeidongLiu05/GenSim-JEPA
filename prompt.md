## 任务目标

你需要在现有 **GenSim-JEPA** 框架基础上，针对 `AzureLoong_simplified.urdf` 定义的人形机器人，配置**行走任务**的训练环境，并**禁用高层策略网络**，仅使用低层 SAC 策略进行端到端控制。

---

## 一、环境配置（Unity ML-Agents）

### 1. 机器人状态观测（与 URDF 对应）

使用以下观测（可参考 `ALGORITHM.md` 中的布局，但需适配人形机器人）：

- **proprio (34维)**：
  - `[0:3]` 躯干线速度 (世界坐标系)
  - `[3:6]` 躯干角速度
  - `[6]` 躯干高度 (y)
  - `[7]` 倾斜角 (度)
  - `[8]` 前向速度 (本地坐标系 z)
  - `[9]` 线加速度大小
  - `[10:22]` 12 个关节位置（按顺序：左右臂各7个？但此URDF有头部、腰部、双腿等，需确认总数）。实际关节数量：头部2个（yaw,pitch），左右臂各7个（共14），腰部3个（pitch,roll,yaw），左右腿各6个（roll,yaw,pitch,knee,ankle_pitch,ankle_roll）共12，总计2+14+3+12=31个关节。但此处 proprio 预留 12 个关节位置和速度，可能只取主要下肢？建议根据实际 Unity 中 Agent 的观测定义调整。此处你只需在 Unity 中定义观测，并在 Python 端解析，不必在此详述。
- **sensor (8维)**：可根据需要自定义，如地面接触、目标方向、速度指令等。
- **vision**：可选，低分辨率图像。

### 2. 动作空间

低层策略输出 10 维原始动作 `a_prim`，然后通过 `ActionDecoder` 映射为 8 维物理动作（关节位置增量或扭矩）。建议动作对应 **双腿和腰部的关键关节**（例如髋、膝、踝，共6个？实际人形机器人行走主要靠双腿，可只控制下肢关节，而上肢保持自然摆动或固定）。为了简化，你可以控制左右腿各 6 个关节（hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll），共12个关节，但动作空间是8维，可能需降维或使用对称映射。建议直接输出 12 个关节目标位置（或速度），不使用 ActionDecoder 压缩，或者修改 ActionDecoder 输出维度。但根据你的架构，`a_prim` 是10维，`ActionDecoder` 输出8维物理动作和持续时间。可能动作空间需适配。最好在 Unity 中定义动作为 12 个关节的扭矩或位置增量，并相应调整 `ActionDecoder` 输出维度。本提示词不限定具体维度，你需根据实际情况调整。

---

## 二、行走任务奖励函数（在 Unity C# Agent 中实现）

设计奖励函数，鼓励机器人向前行走并保持平衡：

```csharp
// 伪代码，实际在 CustomGewuAgent.cs 中
float GetReward()
{
    float forwardVel = Vector3.Dot(robotBody.velocity, transform.forward);
    float height = robotBody.position.y;
    float tilt = Vector3.Angle(transform.up, Vector3.up);
    float energy = jointVelocities.Sum(v => v * v); // 惩罚能耗
    float footContact = leftFootContact && rightFootContact ? 0.5f : 1.0f; // 鼓励交替触地

    float reward = 0f;
    // 主要目标：前向速度
    reward += 1.0f * forwardVel;  // 单位 m/s，可缩放
    // 姿态奖励
    reward -= 0.5f * tilt / 30f;  // 倾斜惩罚
    reward += 0.2f * (height > 0.5f ? 1 : 0); // 站立高度
    // 能耗惩罚
    reward -= 0.01f * energy;
    // 步态奖励：交替触地
    reward += 0.1f * footContact;
    // 存活奖励
    reward += 0.01f;
    // 终止条件：跌倒（高度<0.3 或 tilt>45°）则给予 -1 并结束
    return reward;
}
```

**终止条件**：跌倒（躯干高度低于 0.3m 或倾斜角 > 45°）或超出最大步数。

---

## 三、禁用高层策略网络（仅使用低层 SAC）

### 修改 `config/gsjepa.yaml`

添加参数 `use_hierarchy: false`，并在阶段一训练时忽略高层网络。

### 修改 `stage1_trainer.py`

在 `__init__` 中读取 `use_hierarchy` 标志：

```python
self.use_hierarchy = config.get('use_hierarchy', True)
```

在 `collect_episode` 中，如果 `self.use_hierarchy` 为 False，则：

- 不再调用 `high_actor` 和 `subgoal_encoder`。
- 固定 `g_target = torch.zeros_like(z)`（或直接不传入子目标，修改 `low_actor` 输入）。
- 不调度 `termination_critic` 和 `GIA`，直接每步执行低层策略，直到 episode 结束。

简化后的 `collect_episode` 大致逻辑：

```python
if not self.use_hierarchy:
    # 固定子目标为零向量
    g_target = torch.zeros_like(z)
    while not done:
        a_prim, _ = self.low_actor(z, g_target)
        a_phys, duration = self.action_decoder(a_prim)
        # 执行动作...
        # 存储 transition，g_target 使用零向量
        self.buffer.add_low(obs, g_target, a_phys, r_total, obs_next, done)
```

同时，`train_low_level` 中 `g_target` 从 buffer 取出，但此时 buffer 存的 `g_target` 是零向量，不影响。

### 修改 `low_actor` 网络

可简化输入，去掉 `g_target`，仅输入 `z`，以提高效率。但也可保留，因为零向量不影响，只是多余。

---

## 四、训练流程

1. **阶段一**：运行端到端预训练（SAC + 辅助自监督损失），收集数据。确保奖励函数能驱动机器人迈步。
2. **阶段二**：使用阶段一数据训练 LeWorldModel，学习行走动力学。
3. **阶段三**（可选）：微调策略，加入世界模型的内在奖励，提升稳定性。

由于行走任务比站立复杂，可能需要较长的训练时间（数万至数十万步）。建议先让机器人在平坦地面训练，逐步增加扰动。

---

## 五、代码修改清单

| 文件 | 修改内容 |
|------|----------|
| `config/gsjepa.yaml` | 添加 `use_hierarchy: false` |
| `stage1_trainer.py` | 在 `collect_episode` 中根据 `use_hierarchy` 分支执行简化逻辑；调整 `low_actor` 输入（可选） |
| `low_level_policy.py` | （可选）移除 `g_target` 输入，只接收 `z`，若保留则需保证 `g_target` 为零向量 |
| `unity_env_wrapper.py` | 确保观测格式与 URDF 一致，并正确解析 |
| `CustomGewuAgent.cs` | 实现上述奖励函数和终止条件 |

---

## 六、注意事项

- 动作空间维度需与 `ActionDecoder` 输出匹配。建议在 Unity 中定义动作为 12 个关节目标位置，并调整 `ActionDecoder` 输出为 12 维。
- 若使用低分辨率视觉，可开启视觉编码器，但行走任务主要依赖 proprio 和 sensor，视觉非必需。
- 训练初期可降低学习率，防止不稳定。
- 使用较大的 replay buffer（如 500k）存储多样化的行走尝试。

---

请根据以上指导修改代码，并在 Unity 中配置相应的 Agent 脚本。训练完成后，您将获得一个能够稳定行走的低层策略。