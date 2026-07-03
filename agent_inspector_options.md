# Inspector 选项说明

## CustomAgentStandup（当前训练用脚本）

`Assets/JEPAScenes/basic/CustomAgentStandup.cs`

### Joint Lock Config（关节锁定配置）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Joint Lock Config` | 列表 | — | 所有 ArticulationBody 的列表。勾选 `locked` 的关节被物理锁定（stiffness=10000），网络输出的对应维度动作被跳过。右键脚本标题 → **Refresh Joints** 自动填充。 |

### ML-Agents Settings（ML-Agents 参数）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Max Step` | int | 2000 | 每轮最大步数，超时自动结束 |
| `Proprio Dim` | int | 10 | 本体感觉基础维度（运行时自动计算为 10 + 2×总关节数） |
| `Sensor Dim` | int | 8 | 外部传感器维度 |

### Curriculum & Disturbance（课程推力干扰）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Enable Disturbance` | bool | true | 启用随机推力干扰，每轮固定间隔向机器人施加随机方向冲量 |
| `Push Interval Range` | Vector2Int | (50, 150) | 推力间隔范围（FixedUpdate 步数） |
| `Push Force Range` | Vector2 | (50, 200) | 推力大小范围（ForceMode.Impulse） |
| `Push Angle Range` | float | 360 | 推力水平方向随机范围（度） |
| `Push Vertical` | bool | false | 推力是否包含随机垂直分量（±30°） |
| `Curriculum Level` | int [0-2] | 0 | 课程等级，每升一级推力 × `Curriculum Force Scale` |
| `Curriculum Force Scale` | float | 1.5 | 每级课程推力倍率 |

### Lift Assist（辅助升力）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Enable Lift Assist` | bool | false | 启用持续向上的辅助升力（世界坐标 Y 轴） |
| `Lift Force` | float | 70 | 升力大小（ForceMode.Force） |

### Joint Drive Settings（关节驱动参数）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Stiffness` | float | 2000 | 关节驱动刚度 |
| `Damping` | float | 100 | 关节驱动阻尼 |
| `Force Limit` | float | 300 | 关节驱动力上限 |

### Episode Start Config（每轮初始状态）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Randomize Start Pos` | bool | false | 每轮随机化初始位置偏移 |
| `Start Pos Range` | float | 1 | 位置偏移范围（±米） |
| `Randomize Start Rot` | bool | false | 每轮随机化初始旋转 |
| `Start Pitch Range` | Vector2 | (-80, 80) | 俯仰随机化范围（度） |
| `Start Yaw Range` | Vector2 | (-180, 180) | 偏航随机化范围（度） |

固定模式（两者均为 false）时，机器人从 `(0, y, 0)` 竖直站立开始。

### Initial Joint Angles（初始关节角度）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Initial Joint Angles Degrees` | float 列表 | — | 每轮开始时设置的各关节角度（度），按关节发现顺序排列。右键脚本标题 → **Refresh Initial Angles** 从当前场景捕获。 |

### Standup Reward Weights（站立奖励权重）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Up Dot Reward Scale` | float | 1.0 | `UpDotVertical` 的乘系数。`UpDot = Dot(body.up, Vector3.up)`，范围 [-1, 1] |
| `Action Energy Penalty` | float | 0.3 | 能耗惩罚系数：`-penalty × Σ│action│`（仅统计未锁定关节） |
| `Up Dot Fall Threshold` | float | 0.7 | 直立度阈值。低于此值时跳过能耗惩罚，直接加 `-FallHardPenalty` |
| `Fall Hard Penalty` | float | 2.0 | 快摔倒时的硬惩罚值 |
| `Survive Reward` | float | 0.01 | 每步存活奖励 |

**奖励公式**（每 FixedUpdate 帧）：
```
reward = UpDotRewardScale × UpDot + SurviveReward
if UpDot >= UpDotFallThreshold:
    reward -= ActionEnergyPenalty × Σ│action│
else:
    reward -= FallHardPenalty
```

### Episode End Config（终止条件）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Fall Angle Threshold` | float | 65 | 欧拉角俯仰/翻滚超过此角度（度）即判定摔倒结束 |

### Foot Contacts（足部接触）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Right Foot` | Transform | — | 右足 Transform，用于射线碰撞检测 |
| `Left Foot` | Transform | — | 左足 Transform |
| `Foot Ray Length` | float | 0.15 | 足部向下射线长度 |
| `Ground Layer` | LayerMask | -1 | 地面层级 |

### Raycast（障碍物检测）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Ray Length` | float | 10 | 障碍物射线长度 |
| `Obstacle Layer` | LayerMask | -1 | 障碍物层级 |

### Action Smoothing（动作平滑）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Action Smooth Factor` | float | 0.5 | 指数平滑系数。`prev = factor × prev + (1-factor) × raw`。0 = 无平滑，越接近 1 越平滑 |

### Visualize（可视化）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Show Action Overlay` | bool | false | 在 Game 窗口左上角显示动作叠加层（关节名、原始值、当前位置） |

### Camera（摄像头）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Camera Object` | GameObject | — | 摄像头 GameObject。非空时自动添加 CameraSensorComponent |
| `Camera Width` | int | 64 | 视觉观测宽度 |
| `Camera Height` | int | 64 | 视觉观测高度 |

### Goal（目标标记）

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Goal Marker` | Transform | — | 目标位置标记。通过 SideChannel 从 Python 接收坐标并更新此标记位置 |

---

## Tinker3Agent（参考脚本，Competition 场景用）

`Assets/Competition/Tinker3Agent.cs`

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `WASD` | bool | false | 使用 WASD 键控制（与方向键二选一） |
| `Keyboard` | bool | false | 启用键盘手动控制模式 |
| `Fixbody` | bool | false | 固定机器人不动（调试用） |
| `Train` | bool | — | 训练模式（影响克隆数量、结束逻辑等） |
| `VR` | float | 0 | 目标前向速度，由 `OnEpisodeBegin` 随机化 |
| `WR` | float | 0 | 目标旋转速度 |
| `CR` | float | 0 | 目标弧度/曲率参数 |
| `Disturb` | bool | false | 启用随机推力干扰（注释未启用） |
| `Lift` | bool | false | 启用向上辅助升力 |
| `FY` | float | 70 | 升力大小 |
| `Observation Num` | int | — | 观测维度（运行时计算 = 9 + 2×ActionNum） |
| `Action Num` | int | — | 动作维度（自动检测） |

**奖励公式**（每 FixedUpdate 帧，最终生效的覆盖版本）：
```
UpDotVertical = Dot(body.up, Vector3.up)

if UpDotVertical >= 0.7:
    u_penalty = -0.3 × Σ│u[i]│
else:
    u_penalty = -2.0

reward = UpDotVertical + u_penalty
```
