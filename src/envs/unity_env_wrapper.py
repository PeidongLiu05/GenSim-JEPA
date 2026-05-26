from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np

class UnityEnvWrapper:
    def __init__(self, env_path=None, worker_id=0, seed=42):
        channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=env_path, worker_id=worker_id, 
                                    side_channels=[channel])
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.action_size = self.spec.action_spec.continuous_size
        self.state_size = self.spec.observation_specs[0].shape[0]  # 假设第一个obs是状态向量
        
    def step(self, action):
        # action: numpy array (continuous)
        self.env.set_actions(self.behavior_name, [action])
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(decision_steps) > 0:
            obs = decision_steps.obs[0]  # 第一个观测
            reward = decision_steps.reward[0]
            done = False
        else:
            obs = terminal_steps.obs[0]
            reward = terminal_steps.reward[0]
            done = True
        return obs, reward, done
    
    def reset(self):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs[0]
    
    def close(self):
        self.env.close()