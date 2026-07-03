"""
Unity ML-Agents 环境交互封装层（支持多智能体并行）。

观测格式（始终保留 batch 维）：
    obs = {
        'proprio': ndarray (n_agents, proprio_dim)
        'sensor':  ndarray (n_agents, sensor_dim)
        'vision':  ndarray (n_agents, C, H, W)
    }
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
import numpy as np
import uuid
from typing import Optional, Tuple


# ======================================================================
#  Goal SideChannel：与 C# GoalSideChannel 镜像
# ======================================================================
class GoalStateChannel(SideChannel):
    """与 Unity C# GoalSideChannel 通信，发送目标位置并接收机器人状态。"""

    def __init__(self):
        super().__init__(uuid.UUID("3f3a7c8e-1d2e-4b0f-9a6c-5d8e2f1a7b3c"))
        self._robot_pos = np.zeros(3)
        self._joint_positions = np.zeros(32)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """接收从 Unity 发来的机器人状态。"""
        self._robot_pos[0] = msg.read_float32()
        self._robot_pos[2] = msg.read_float32()
        for i in range(len(self._joint_positions)):
            self._joint_positions[i] = msg.read_float32()

    def _write_type(self, msg: OutgoingMessage, type_id: int) -> None:
        """写入消息类型头 (write_int32 以匹配 C# ReadInt32)。"""
        msg.write_int32(type_id)

    def send_goal(self, goal_x: float, goal_z: float) -> None:
        """发送目标位置到 Unity。"""
        msg = OutgoingMessage()
        self._write_type(msg, 0)
        msg.write_float32(goal_x)
        msg.write_float32(goal_z)
        super().queue_message_to_send(msg)

    def send_curriculum_level(self, level: int) -> None:
        """设置课程等级 (0=简单, 1=中等, 2=困难)。"""
        msg = OutgoingMessage()
        self._write_type(msg, 1)
        msg.write_int32(level)
        super().queue_message_to_send(msg)

    def send_disturbance_enabled(self, enabled: bool) -> None:
        """启用/禁用推力干扰。"""
        msg = OutgoingMessage()
        self._write_type(msg, 2)
        msg.write_int32(1 if enabled else 0)
        super().queue_message_to_send(msg)

    def send_push_force_range(self, min_force: float, max_force: float) -> None:
        """设置推力范围。"""
        msg = OutgoingMessage()
        self._write_type(msg, 3)
        msg.write_float32(min_force)
        msg.write_float32(max_force)
        super().queue_message_to_send(msg)

    @property
    def robot_position(self) -> np.ndarray:
        return self._robot_pos.copy()

    @property
    def joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()


# ======================================================================
#  Unity 环境封装（多智能体支持）
# ======================================================================
class UnityEnvWrapper:
    """
    封装 Unity ML-Agents 环境，提供多智能体类 Gym 接口。

    所有观测/奖励/done 均保留 batch 维 (n_agents, ...)。
    调用方通过 obs['proprio'].shape[0] 获取当前智能体数量。
    """

    def __init__(self,
                 env_path: Optional[str] = None,
                 worker_id: int = 0,
                 seed: int = 42,
                 proprio_dim: int = 10,
                 sensor_dim: int = 38,
                 camera_height: int = 64,
                 camera_width: int = 64,
                 no_graphics: bool = False,
                 timeout_wait: int = 60):
        self.proprio_dim = proprio_dim
        self.sensor_dim = sensor_dim
        self.vec_obs_dim = proprio_dim + sensor_dim
        self.camera_shape = (3, camera_height, camera_width)
        self.n_agents = 1

        self.goal_channel = GoalStateChannel()
        engine_channel = EngineConfigurationChannel()

        self.env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            side_channels=[engine_channel, self.goal_channel],
            no_graphics=no_graphics,
            timeout_wait=timeout_wait,
        )
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        num_obs = len(self.spec.observation_specs)
        print(f"[UnityEnv] 行为名称: {self.behavior_name}")
        print(f"[UnityEnv] 观测数量: {num_obs}")
        for i, spec in enumerate(self.spec.observation_specs):
            actual_shape = spec.shape[1:] if len(spec.shape) > 1 else spec.shape
            print(f"  obs[{i}]: shape={actual_shape}, type={spec.observation_type}")
        cont_size = self.spec.action_spec.continuous_size
        discrete_branches = self.spec.action_spec.discrete_branches
        print(f"[UnityEnv] 连续动作维度: {cont_size}, 离散分支: {discrete_branches}")
        print(f"[UnityEnv] 设定的 proprio_dim={proprio_dim} sensor_dim={sensor_dim}")

    # ------------------------------------------------------------------
    #  观测解析（保留 batch 维）
    # ------------------------------------------------------------------
    def _parse_obs(self, obs_list) -> dict:
        """
        解析 obs_list 为 {proprio, sensor, vision}，均保留 (n_agents, ...) 形状。
        obs_list 中的每个元素形状为 (n_agents, *obs_shape)。
        """
        vec = None
        vision = None
        for obs in obs_list:
            arr = np.array(obs, dtype=np.float32)
            if arr.ndim == 2:          # (n_agents, vec_dim)
                vec = arr
            elif arr.ndim == 4:        # (n_agents, C, H, W)
                vision = arr
            elif arr.ndim == 3:        # (n_agents, H, W) 灰度
                vision = arr

        n = vec.shape[0] if vec is not None else (vision.shape[0] if vision is not None else 1)

        if vec is None:
            vec = np.zeros((n, self.vec_obs_dim), dtype=np.float32)
        elif vec.shape[1] < self.vec_obs_dim:
            padded = np.zeros((n, self.vec_obs_dim), dtype=np.float32)
            padded[:, :vec.shape[1]] = vec
            vec = padded

        proprio = vec[:, :self.proprio_dim]
        sensor = vec[:, self.proprio_dim:self.proprio_dim + self.sensor_dim]

        if vision is None:
            vision = np.zeros((n, *self.camera_shape), dtype=np.float32)
        else:
            # 灰度 (n, H, W) → (n, 1, H, W)
            if vision.ndim == 3:
                vision = vision[:, np.newaxis, :, :]
            # HWC → CHW
            if vision.shape[-1] in (1, 3):
                vision = np.transpose(vision, (0, 3, 1, 2))
            # 单通道 → 3 通道
            if vision.shape[1] == 1:
                vision = np.repeat(vision, 3, axis=1)
            vision = vision / 255.0 if vision.max() > 1.0 else vision

        return {'proprio': proprio, 'sensor': sensor, 'vision': vision}

    # ------------------------------------------------------------------
    #  重置
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """重置环境，返回初始观测 (n_agents, ...)。"""
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.n_agents = len(decision_steps)
        return self._parse_obs(decision_steps.obs)

    # ------------------------------------------------------------------
    #  单步推进（多智能体）
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        执行一步动作。

        参数:
            action: (n_agents, action_dim)  — 所有智能体的动作

        返回:
            obs:    dict {proprio, sensor, vision}, 各数组形状 (n_agents, ...)
            reward: (n_agents,)  — 每个智能体的奖励
            done:   (n_agents,)  — True 表示该智能体已终止
        """
        if action.ndim == 1:
            action = action[np.newaxis, :]

        self.env.set_actions(self.behavior_name, ActionTuple(continuous=action))
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        n = self.n_agents

        # 初始化全 done
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.ones(n, dtype=bool)
        all_obs = None

        def _fill_obs(target, source_dict, ids):
            """将 source_dict 按 ids 填入 target 对应位置。"""
            if target is None:
                target = {k: np.zeros((n, *v.shape[1:]), dtype=v.dtype) for k, v in source_dict.items()}
            for k, v in source_dict.items():
                target[k][ids] = v
            return target

        if len(decision_steps) > 0:
            alive_ids = decision_steps.agent_id
            alive_obs = self._parse_obs(decision_steps.obs)
            rewards[alive_ids] = decision_steps.reward
            dones[alive_ids] = False
            all_obs = _fill_obs(all_obs, alive_obs, alive_ids)

        if len(terminal_steps) > 0:
            dead_ids = terminal_steps.agent_id
            dead_obs = self._parse_obs(terminal_steps.obs)
            rewards[dead_ids] = terminal_steps.reward
            dones[dead_ids] = True
            all_obs = _fill_obs(all_obs, dead_obs, dead_ids)

        if all_obs is None:
            all_obs = {
                'proprio': np.zeros((n, self.proprio_dim), dtype=np.float32),
                'sensor': np.zeros((n, self.sensor_dim), dtype=np.float32),
                'vision': np.zeros((n, *self.camera_shape), dtype=np.float32),
            }

        return all_obs, rewards, dones

    # ------------------------------------------------------------------
    #  （以下用于分层模式，多智能体场景下暂未适配）
    # ------------------------------------------------------------------
    def get_goal_state(self) -> np.ndarray:
        return np.zeros(self.proprio_dim, dtype=np.float32)

    def request_goal(self) -> np.ndarray:
        goal_x = np.random.uniform(-8.0, 8.0)
        goal_z = np.random.uniform(-8.0, 8.0)
        self.goal_channel.send_goal(goal_x, goal_z)
        goal_state = np.zeros(self.proprio_dim, dtype=np.float32)
        goal_state[0] = goal_x * 0.1
        goal_state[1] = goal_z * 0.1
        return goal_state

    def close(self):
        self.env.close()
