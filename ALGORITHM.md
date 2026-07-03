# GenSim-JEPA 算法框架

分层强化学习 + 联合嵌入预测架构（JEPA），用于 Unity ML-Agents 具身智能体控制。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        三维训练流水线                             │
│                                                                  │
│  阶段一                    阶段二                     阶段三       │
│ ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│ │ 端到端预训练   │ ──► │ 世界模型训练   │ ──► │ 完整强化学习  │       │
│ │              │     │              │     │              │       │
│ │ SAC + TD3    │     │ LeWMPredictor│     │ 冻结世界模型  │       │
│ │ + 自监督损失  │     │ + SIGReg     │     │ + 内在奖励    │       │
│ └──────────────┘     └──────────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

| 阶段 | 目标 | 核心算法 |
|------|------|----------|
| **阶段一** | 预训练所有编码器/解码器/策略网络 | SAC (低层) + TD3 (高层) + 自监督辅助损失 |
| **阶段二** | 训练前向动力学预测器 (世界模型) | JEPA 式隐空间预测 + SIGReg 正则化 |
| **阶段三** | 基于世界模型的完整 RL 微调 | 冻结世界模型提供内在奖励 + 自监督 GIA 训练 |

---

## 二、网络架构总览

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  观测输入                        ┌───────────────────┐                    │
│  ┌──────┐  ┌──────┐  ┌──────┐   │   StateEncoder    │                    │
│  │proprio│  │sensor│  │vision│    │                   │                    │
│  │ (72)  │  │ (8)  │(3,64,64)   │  ┌─────────────┐  │   z (128)          │
│  └──┬───┘  └──┬───┘  └──┬───┘   │  │proprio_enc  │──┤  ┌─────────────┐  │
│     │         │         │       │  │ Linear(72→64)│  │  │             │  │
│     │         │         │       │  └──────┬──────┘  │  │  LeWMPredict│  │
│     │         │         │       │         │         │  │  or (JEPA)  │  │
│     │         │         │       │  ┌─────────────┐  │  │             │  │
│     └─────────┼─────────┼───────┼──┤sensor_enc   │──┼──┤ z_t + a_prim│  │
│               │         │       │  │ Linear(8→64) │  │  │    ↓        │  │
│               │         │       │  └──────┬──────┘  │  │ Transformer │  │
│               │         │       │         │         │  │    ↓        │  │
│               │         │       │  ┌─────────────┐  │  │ z_{t+1}_pred│  │
│               │         │       │  │vision_enc   │──┼──┤  (128)      │  │
│               │         │       │  │(ViT patch→  │  │  └─────────────┘  │
│               │         │       │  │ mean pool)  │  │                   │
│               │         │       │  └─────────────┘  │  阶段二/三冻结     │
│               │         │       └────────┬──────────┘                   │
│               │         │                │ z (128)                      │
│               │         │                ▼                              │
│               │         │     ┌──────────────────┐                      │
│               │         │     │  高層策略 (TD3)   │                      │
│               │         │     │                  │                      │
│               │         │     │ HighLevelPolicy  │── g (128)            │
│               │         │     │ cat(z, z_goal)   │                      │
│               │         │     └──────────────────┘                      │
│               │         │                │                              │
│               │         │                ▼                              │
│               │         │     ┌──────────────────┐                      │
│               │         │     │  SubgoalEncoder   │── g_target (128)    │
│               │         │     │  Linear(128→256→128)                    │
│               │         │     └──────────────────┘                      │
│               │         │                │                              │
│               │         │                ▼                              │
│               │         │     ┌──────────────────┐                      │
│               │         │     │  低層策略 (SAC)   │                      │
│               │         │     │                  │                      │
│               │         │     │ LowLevelPolicy   │── a_prim (10)        │
│               │         │     │ Normal(μ, σ)     │  + log_prob          │
│               │         │     │ ┌──────────────┐ │                      │
│               │         │     │ │    tanh      │ │                      │
│               │         │     │ └──────────────┘ │                      │
│               │         │     └──────────────────┘                      │
│               │         │                │                              │
│               │         │                ▼                              │
│  │         │     ┌──────────────────┐                      │
│  │         │     │  ActionDecoder    │── a_phys (31)       │
│  │         │     │  Linear(10→64→32)│  + duration (1)     │
│               │         │     └──────────────────┘                      │
│               │         │                │                              │
│               │         │                ▼                              │
│               │         │         Unity 环境                            │
│               │         │     ┌──────────────────┐                      │
│               │         │     │   Termination    │── prob ∈ [0,1]      │
│               │         │     │   Critic         │  是否终止当前子目标    │
│               │         │     └──────────────────┘                      │
│               │         │     ┌──────────────────┐                      │
│               │         │     │   GIA            │── 预测距离            │
│               │         │     │   GoalInfoAcquirer│  停滞检测用           │
│               │         │     └──────────────────┘                      │
└───────────────┴─────────┴──────────────────────────────────────────────┘
```

### 维度一览

| 网络 | 输入 | 输出 |
|------|------|------|
| **StateEncoder** | proprio(72) + sensor(8) + vision(3,64,64) | z(128) |
| **ActionEncoder** | a_phys(31) + duration(1) | a_prim(10) |
| **ActionDecoder** | a_prim(10) | a_phys(31) + duration(1) |
| **GoalEncoder** | goal_state(72) | z_goal(128) |
| **SubgoalEncoder** | g(128) | g_target(128) |
| **LowLevelPolicy (SAC)** | z(128) + g_target(128) | a_prim(10) + log_prob |
| **LowLevelCritic** | z(128) + g_target(128) + a(10) | Q(1) |
| **HighLevelPolicy (TD3)** | z(128) + z_goal(128) | g(128) |
| **HighLevelCritic** | z(128) + z_goal(128) + g(128) | Q(1) |
| **TerminationCritic** | z(128) + g_target(128) | prob(1) |
| **GoalInfoAcquirer (GIA)** | z(128) + z_goal(128) + g_target(128) + uncert(1) | dist(1) |
| **LeWMPredictor** | z(128) + a_prim(10) | z_pred(128) |

---

## 三、阶段一：端到端预训练

### 损失函数

```
总损失 = 低层Critic损失₁ + 低层Critic损失₂
       + 低层Actor损失 (SAC)
       + 自适应熵损失
       + λ_recon  × 状态重构损失
       + λ_dyn   × 前向动态损失
       + λ_cycle × 动作循环一致性损失
```

### 辅助自监督学习

```
                ┌──────────────────────────────┐
                │        状态重构               │
                │                              │
                │  z ──► state_decoder ──► [proprio, sensor]  │
                │                              │
                │  Loss = MSE(重构, 真实)       │
                └──────────────────────────────┘

                ┌──────────────────────────────┐
                │      前向动态预测             │
                │                              │
                │  [z, z_a] ──► forward_dyn ──► z_next_pred    │
                │                              │
                │  Loss = MSE(z_next_pred, z_next)             │
                └──────────────────────────────┘

                ┌──────────────────────────────┐
                │    动作循环一致性              │
                │                              │
                │  a_phys ──► ActionEncoder ──► a_prim         │
                │  a_prim ──► ActionDecoder ──► a_phys'        │
                │                              │
                │  Loss = MSE(a_prim, a_prim')                 │
                │       + MSE(a_phys, a_phys')                 │
                └──────────────────────────────┘
```

### 子目标重标定 (HIRO 简化版)

```
每轮高層经验收集后:
    g_new = z_next - z   ▷ 实际到达的状态变化量作为新子目标
    替代原有 g_old 存入 buffer
```

---

## 四、阶段二：世界模型训练

### JEPA 式预测

```
不是预测像素，而是在隐空间中预测:

    z_t ────┐
             ├──► Transformer ──► z_{t+1}_pred ──► MSE ──► z_{t+1}
    a_t ────┘

    Loss = MSE(z_pred, z_real) + λ_sigreg × SIGReg(z)
```

### SIGReg 正则化

```
迫使隐状态分布接近各向同性高斯 N(0, I):

    1. 随机投影矩阵 A (128 × num_proj)
    2. 计算每个投影的经验特征函数 φ(t) = mean(cos(t·x)) + i·mean(sin(t·x))
    3. 与高斯真实特征函数 φ_g(t) = exp(-t²/2) 比较
    4. Epps-Pulley 统计量作为正则化损失
```

---

## 五、阶段三：基于世界模型的 RL

### 新增内在奖励

```
r_total = r_extrinsic + λ_wm × r_intrinsic

r_intrinsic = -||z_next - LeWMPredictor(z, a_prim)||₂

含义: 世界模型预测不准的状态 → 高内在奖励 → 鼓励探索
```

### Termination Critic 监督信号

```
actual_progress   = ||z_end - z_start||₂
ideal_progress    = rollout(k步, 用世界模型+低层策略确定性模式)

label = 1  if actual_progress < α × ideal_progress  else 0
         (进展太慢,应终止)          (进展正常,继续)
```

### GIA 自监督训练

```
true_dist = rollout(horizon步, 世界模型+低层策略)
            └─► 返回 ||z_end - z_goal||₂

Loss = MSE(GIA(z, z_goal, g_target, uncertainty), true_dist)
```

### 推理时停滞检测

```
┌──────────────────────────────────────────────────┐
│  滑动窗口 (10步)                                 │
│                                                  │
│  早期平均 = mean(dist[0:5])                      │
│  近期平均 = mean(dist[5:10])                     │
│                                                  │
│  若 近期平均 ≥ 早期平均:                          │
│      └─► 判定停滞, 请求新目标                     │
│          env.request_goal() ──► 更新 z_goal       │
│          高層重新规划子目标                        │
│          清空滑动窗口                              │
└──────────────────────────────────────────────────┘
```

---

## 六、环境交互数据流

```
┌──────────────────┐    动作 (8维)     ┌──────────────────┐
│                  │ ◄────────────── │                  │
│   Python 端      │    动作 (31维)    │   Unity 端       │
│   GenSim-JEPA    │                  │   ML-Agents      │
│                  │ ──────────────► │                  │
│                  │   观测 + 奖励    │                  │
└──────────────────┘                  └──────────────────┘
         │                                   │
         │ 观测解析                           │
         ▼                                   ▼
   ┌──────────────┐                  ┌──────────────┐
    │ {'proprio'   │                  │ 机器人物理    │
    │  (72,)}       │                  │              │
    │ {'sensor'    │                  │ 31个关节     │
    │  (8,)}        │                  │ (AzureLoong) │
   │ {'vision'    │                  │              │
   │  (3,64,64)}   │                  │ CameraSensor │
   └──────────────┘                  └──────────────┘
         │                                   │
         │ SideChannel                        │
         │ GoalStateChannel                  │ GoalSideChannel
         │ ────────── 目标坐标 ──────────────► │
         │ ◄────────── 机器人位置 ─────────── │
```

### 观测布局 (C# → Python)

```
proprio (72 维, N = totalJointCount = 31):
  [0:3]     = rootBody.velocity (世界空间线速度)
  [3:6]     = rootBody.angularVelocity
  [6]       = body height (y)
  [7]       = tiltAngle (度)
  [8]       = 前向速度 (本地坐标系 z)
  [9]       = 线加速度大小
  [10:10+N) = N 个关节位置 (按各自 xDrive 限位归一化到 ≈[-1,1])
  [10+N:10+2N) = N 个关节速度 (归一化到 ≈[-1,1])

sensor (8 维):
  [0]     = groundContact (1=触地)
  [1]     = bodyHeightNorm
   [2]     = bodyHeightNorm (高度归一化)
   [3]     = obstacleDist (最近障碍物距离)
   [4]     = goalDirX (目标方向, 机器人本地系)
   [5]     = goalDirZ
   [6]     = pitch (俯仰角, 归一化)
   [7]     = roll (翻滚角, 归一化)

vision: CameraSensorComponent → 64×64 RGB, 归一化到 [0,1]
```

---

## 七、文件结构

```
GenSim-JEPA/
│
├── train.py                          # 训练入口 (三阶段)
├── inference.py                      # 推理入口
├── ALGORITHM.md                      # 本文件
│
├── configs/
│   └── train/models/
│       └── gsjepa.yaml               # 全局超参数配置
│
├── envs/
│   └── unity_env_wrapper.py          # Unity 环境封装
│
├── trainers/
│   ├── stage1_trainer.py             # 阶段一: SAC + TD3 + 自监督
│   ├── stage2_trainer.py             # 阶段二: JEPA 世界模型
│   └── stage3_trainer.py             # 阶段三: 完整 RL 微调
│
├── networks/
│   ├── state_encoder.py              # 多模态状态编码器
│   ├── action_encoder.py             # 动作编码器 (物理→隐)
│   ├── action_decoder.py             # 动作解码器 (隐→物理)
│   ├── goal_encoder.py               # 目标编码器
│   ├── subgoal_encoder.py            # 子目标编码器
│   ├── low_level_policy.py           # 低层策略 (SAC)
│   ├── low_level_critic.py           # 低层 Q 网络
│   ├── high_level_policy.py          # 高层策略 (TD3)
│   ├── high_level_critic.py          # 高层 Q 网络
│   ├── termination_critic.py         # 子目标终止判别器
│   ├── goal_info_acquirer.py         # GIA: 目标距离预测
│   ├── lewm_predictor.py             # LeWorldModel 预测器
│   └── utils/
│       ├── vision_transformer.py     # ViT 视觉编码器
│       └── sigreg.py                 # SIGReg 正则化
│
├── buffers/
│   └── hierarchical_buffer.py        # 双层经验回放缓冲区
│
└── 对应 Unity 项目:
gewu/Assets/JEPAScenes/basic/
    └── CustomAgentStandup.cs          # Agent C# 脚本
```

---

## 八、使用方式

### 前置条件

1. **打开 Unity Editor**，按下 Play 按钮（`env_path=None` 模式自动连接当前 Editor 实例）
2. **Inspector 配置**：将 `rightFoot` / `leftFoot` / `cameraObject` / `goalMarker` 拖入对应字段
3. **配置文件**：修改 `configs/train/models/gsjepa.yaml` 中的 `use_hierarchy`：
   - `false`（默认）：仅低层 SAC 策略（适用于行走任务）
   - `true`：完整分层 RL（高层 TD3 + 低层 SAC + GIA）

### 训练

```bash
# 阶段一: 端到端预训练
python train.py --config configs/train/models/gsjepa.yaml --env_path None --stage 1

# 阶段二: 世界模型训练（需阶段一填满 buffer 后再执行）
python train.py --config configs/train/models/gsjepa.yaml --stage 2

# 阶段三: 完整 RL 微调
python train.py --config configs/train/models/gsjepa.yaml --env_path None --stage 3
```

### 推理

```bash
# 非分层模式（行走任务，默认）
python inference.py --checkpoint_dir ./checkpoints

# 分层模式
python inference.py --checkpoint_dir ./checkpoints --config configs/train/models/gsjepa.yaml
```

### 续接中断的训练

训练意外中断后，可以使用 `--resume` 从指定 checkpoint 恢复：

```bash
# 阶段一续接
python train.py --config configs/train/models/gsjepa.yaml --stage 1 --resume checkpoints/stage1_nohier_ep500.pt --resume_ep 500

# 阶段二续接
python train.py --config configs/train/models/gsjepa.yaml --stage 2 \
    --resume checkpoints/stage2_epoch200.pt

# 阶段三续接
python train.py --config configs/train/models/gsjepa.yaml --stage 3 \
    --resume checkpoints/stage3_nohier_ep5000.pt --resume_ep 5000
```

- `--resume checkpoint.pt`：加载网络权重和优化器状态
- `--resume_ep N`：设置起始 episode 编号（覆盖 checkpoint 中的 ep）
- checkpoint 自动保存在 `checkpoints/` 目录，命名格式为 `stage{1,2,3}_{hier|nohier}_ep{N}.pt`
- 维度变更（如修改 proprio_dim/action_dim）后，旧 checkpoint 不兼容，需重新训练

### 模型权重文件

| 模式 | 阶段一保存 | 推理加载 |
|------|-----------|---------|
| `use_hierarchy: false` | `checkpoints/stage1_nohier_ep{ep}.pt` | `stage2_state_encoder.pth` + `stage3_low_actor.pth` + `stage3_action_decoder.pth` |
| `use_hierarchy: true` | `checkpoints/stage1_hier_ep{ep}.pt` | 同上 + `stage3_high_actor.pth` + `stage3_termination_critic.pth` + `stage3_gia.pth` |

注意：阶段一保存的是字典格式 `.pt`，推理需要的是单独 `.pth` 文件。
阶段一训练完成后，需提取权重：

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/stage1_nohier_ep50000.pt')
torch.save(ckpt['state_encoder'], 'checkpoints/stage2_state_encoder.pth')
torch.save(ckpt['action_decoder'], 'checkpoints/stage3_action_decoder.pth')
torch.save(ckpt['low_actor'], 'checkpoints/stage3_low_actor.pth')
"

---

## 九、算法谱系

```
                         ┌──── DQN ──── DDPG ──── TD3 ──────────┐
                         │                          (高层策略)    │
    Model-Free RL ───────┤                                        ├──► HRL
                         │                          (低层策略)    │      (本框架)
                         └──── Policy Gradient ── SAC ───────────┘
                                    ↑
                                    │
                                JEPA ───────────────► LeWorldModel
                                (联合嵌入预测架构)     (隐空间预测 + SIGReg)
                                    ↑
                                    │
                              I-JEPA (Meta 2023)
```

核心设计思路：**在隐空间中做预测**（JEPA），而非在像素空间，从而避免重建高维视觉的困难，同时保留足够的信息用于 RL 决策。

---

## 十、非分层模式（行走任务默认）

当 `use_hierarchy: false` 时，跳过高层策略、GIA、TerminationCritic 等模块，仅运行低层 SAC：

```
                              ┌──────────┐
  obs ──► StateEncoder ──► z ─► LowLevelPolicy ──► a_prim ──► ActionDecoder ──► a_phys (31)
                              │                      ↑
                              │   g_target = zeros    │
                              └──────────────────────┘

  简化后的损失: 低层Critic损失₁ + 低层Critic损失₂ + 低层Actor损失 + 自适应熵
              + λ_recon × 状态重构损失 + λ_dyn × 前向动态损失 + λ_cycle × 动作循环一致性
```

- `g_target` 固定为零向量（不做子目标追踪）
- 推理时只加载 `state_encoder`、`low_actor`、`action_decoder` 三个网络
- 适用于纯行走任务，不依赖目标导向探索
