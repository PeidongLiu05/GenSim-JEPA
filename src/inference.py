import torch
from networks import *

def inference():
    # 加载预训练权重
    state_encoder = StateEncoder()
    state_encoder.load_state_dict(torch.load("state_encoder.pth"))
    state_encoder.eval()
    
    goal_encoder = GoalEncoder()
    goal_encoder.load_state_dict(torch.load("goal_encoder.pth"))
    goal_encoder.eval()
    
    high_policy = HighLevelPolicy()
    high_policy.load_state_dict(torch.load("high_policy.pth"))
    high_policy.eval()
    
    subgoal_encoder = SubgoalEncoder()
    subgoal_encoder.load_state_dict(torch.load("subgoal_encoder.pth"))
    subgoal_encoder.eval()
    
    low_policy = LowLevelPolicy()
    low_policy.load_state_dict(torch.load("low_policy.pth"))
    low_policy.eval()
    
    action_decoder = ActionDecoder()
    action_decoder.load_state_dict(torch.load("action_decoder.pth"))
    action_decoder.eval()
    
    termination_critic = TerminationCritic()
    termination_critic.load_state_dict(torch.load("termination_critic.pth"))
    termination_critic.eval()
    
    # 连接真实机器人或Unity环境
    env = UnityEnvWrapper("path_to_executable")  # 推理环境
    
    obs = env.reset()
    z = state_encoder(obs)
    final_goal = goal_encoder(goal_state)  # 任务终点
    
    while True:
        g = high_policy(z, final_goal)
        g_target = subgoal_encoder(g)
        # 执行子目标直到调度器判定终止
        while True:
            dist = low_policy(z, g_target, deterministic=True)
            a_prim = dist[0]  # 确定性输出
            a_phys, duration = action_decoder(a_prim)
            # 执行动作，获取新obs
            next_obs, reward, done = env.step(a_phys.detach().numpy())
            z = state_encoder(next_obs)
            term_prob = termination_critic(z, g_target)
            if term_prob > 0.5 or done:
                break
        if done:
            break