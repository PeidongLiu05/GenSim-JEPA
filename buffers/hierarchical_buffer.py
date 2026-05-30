from collections import deque
import numpy as np
import random
import torch

class HierarchicalReplayBuffer:
    def __init__(self, high_capacity=100000, low_capacity=100000):
        self.high_buffer = deque(maxlen=high_capacity)
        self.low_buffer = deque(maxlen=low_capacity)

    def add_low(self, obs, g, a_phys, r, obs_next, done):
        self.low_buffer.append((obs, g, a_phys, r, obs_next, done))

    def add_high(self, z, z_goal, g, sum_r, z_next, done, low_transitions):
        self.high_buffer.append((z, z_goal, g, sum_r, z_next, done, low_transitions))

    def sample_low(self, batch_size):
        batch = random.sample(self.low_buffer, batch_size)
        obs, g, a_phys, r, obs_next, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(obs), torch.FloatTensor(g), torch.FloatTensor(a_phys),
                torch.FloatTensor(r), torch.FloatTensor(obs_next), torch.FloatTensor(done))

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
        self.high_buffer = deque(new_high_buffer, maxlen=self.capacity)