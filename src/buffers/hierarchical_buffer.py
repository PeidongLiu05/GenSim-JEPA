import torch
import numpy as np
from collections import deque
import random

class HierarchicalReplayBuffer:
    """
    存储高层和低层经验，支持subgoal relabeling
    """
    def __init__(self, capacity=100000, low_capacity=100000):
        self.capacity = capacity
        self.low_capacity = low_capacity
        self.high_buffer = deque(maxlen=capacity)
        self.low_buffer = deque(maxlen=low_capacity)
        
    def add_low(self, z, g_target, a_prim, r, z_next, done):
        self.low_buffer.append((z, g_target, a_prim, r, z_next, done))
        
    def add_high(self, z, z_goal, g, z_next, reward, done, low_transitions):
        # low_transitions: list of (z, a_prim, ...) for k steps
        self.high_buffer.append((z, z_goal, g, z_next, reward, done, low_transitions))
        
    def sample_low(self, batch_size):
        batch = random.sample(self.low_buffer, batch_size)
        z, g, a, r, z_next, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(z), torch.FloatTensor(g), 
                torch.FloatTensor(a), torch.FloatTensor(r), 
                torch.FloatTensor(z_next), torch.FloatTensor(done))
    
    def sample_high(self, batch_size):
        batch = random.sample(self.high_buffer, batch_size)
        z, z_goal, g, z_next, r, done, trans = map(list, zip(*batch))
        return (torch.FloatTensor(np.stack(z)), 
                torch.FloatTensor(np.stack(z_goal)),
                torch.FloatTensor(np.stack(g)),
                torch.FloatTensor(np.stack(z_next)),
                torch.FloatTensor(np.stack(r)),
                torch.FloatTensor(np.stack(done)),
                trans)
    
    def relabel_subgoals(self, transition_predictor):
        """
        对high_buffer中的每个经验，重新计算子目标g' = z_{t+k} - z_t
        并更新存储的g
        """
        new_buffer = []
        for (z, z_goal, g_old, z_next, reward, done, low_trans) in self.high_buffer:
            # 实际到达的隐状态差
            g_new = z_next - z
            new_buffer.append((z, z_goal, g_new, z_next, reward, done, low_trans))
        self.high_buffer = deque(new_buffer, maxlen=self.capacity)