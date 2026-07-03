from collections import deque
import numpy as np
import random
import torch

class HierarchicalReplayBuffer:
    def __init__(self, high_capacity=100000, low_capacity=100000):
        self.high_capacity = high_capacity
        self.low_capacity = low_capacity
        self.high_buffer = deque(maxlen=high_capacity)
        self.low_buffer = deque(maxlen=low_capacity)

    def add_low(self, obs, g, a_phys, r, obs_next, done):
        self.low_buffer.append((obs, g, a_phys, r, obs_next, done))

    def add_high(self, z, z_goal, g, sum_r, z_next, done, low_transitions):
        self.high_buffer.append((z, z_goal, g, sum_r, z_next, done, low_transitions))

    def sample_low(self, batch_size):
        batch = random.sample(self.low_buffer, batch_size)
        # batch元素可能是 (proprio,sensor,vision) 元组，也可能是平坦向量
        raw_obs, g, a_phys, r, raw_obs_next, done = zip(*batch)
        if isinstance(raw_obs[0], tuple):
            proprio, sensor, vision = zip(*raw_obs)
            proprio = torch.FloatTensor(np.stack(proprio))
            sensor = torch.FloatTensor(np.stack(sensor))
            vision = torch.FloatTensor(np.stack(vision))
            obs = (proprio, sensor, vision)
            proprio_n, sensor_n, vision_n = zip(*raw_obs_next)
            obs_next = (torch.FloatTensor(np.stack(proprio_n)),
                        torch.FloatTensor(np.stack(sensor_n)),
                        torch.FloatTensor(np.stack(vision_n)))
        else:
            obs = torch.FloatTensor(np.stack(raw_obs))
            obs_next = torch.FloatTensor(np.stack(raw_obs_next))
        r_flat = torch.FloatTensor(np.stack(r))
        done_flat = torch.FloatTensor(np.stack(done))
        if r_flat.dim() > 1:
            r_flat = r_flat.squeeze(-1)
        if done_flat.dim() > 1:
            done_flat = done_flat.squeeze(-1)
        return (obs, torch.FloatTensor(np.stack(g)), torch.FloatTensor(np.stack(a_phys)),
                r_flat, obs_next, done_flat)

    def sample_high(self, batch_size):
        batch = random.sample(self.high_buffer, batch_size)
        z, z_goal, g, r, z_next, done, trans = map(list, zip(*batch))
        return (torch.FloatTensor(np.stack(z)), torch.FloatTensor(np.stack(z_goal)),
                torch.FloatTensor(np.stack(g)), torch.FloatTensor(np.stack(r)),
                torch.FloatTensor(np.stack(z_next)), torch.FloatTensor(np.stack(done)), trans)

    def relabel_subgoals(self, low_policy, k):
        """
        重标定高层经验中的子目标 g，使用 HIRO 的候选目标集方法。
        这里简化为：对于每条高层经验，实际到达的状态差为 z_next - z，
        我们将原 g 替换为该差值，并存储。
        更精确的 HIRO 实现需要评估多个候选目标，此处简化。
        """
        new_high_buffer = []
        for (z, z_goal, g_old, sum_r, z_next, done, low_trans) in self.high_buffer:
            g_new = z_next - z
            new_high_buffer.append((z, z_goal, g_new, sum_r, z_next, done, low_trans))
        self.high_buffer = deque(new_high_buffer, maxlen=self.high_capacity)