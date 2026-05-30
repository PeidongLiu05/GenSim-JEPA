"""
推理脚本：纯深度学习，零 RL，无 CEM
复用阶段三训练好的双层策略网络、termination_critic 和 GIA，
实现长短周期规划与自主目标更新。
"""

import torch
import numpy as np
from collections import deque
from pathlib import Path

# ---------- 辅助函数：观测解析 ----------
def parse_obs(obs):
    """
    将环境观测分解为 proprio, sensor, vision。
    根据实际环境格式实现，假设 obs 为字典。
    """
    proprio = torch.from_numpy(obs['proprio']).float()
    sensor = torch.from_numpy(obs['sensor']).float()
    vision = torch.from_numpy(obs['vision']).float()   # (C, H, W)
    return proprio, sensor, vision


# ---------- 推理主循环 ----------
def inference_loop(config):
    device = torch.device(config['device'])

    # -------------------- 加载所有模型（冻结） --------------------
    # 状态编码器（阶段二）
    state_encoder = torch.load(config['state_encoder_path'], map_location=device)
    state_encoder.eval()
    for p in state_encoder.parameters():
        p.requires_grad = False

    # LeWorldModel 预测器（阶段二）
    lewm_predictor = torch.load(config['predictor_path'], map_location=device)
    lewm_predictor.eval()
    for p in lewm_predictor.parameters():
        p.requires_grad = False

    # 目标编码器（阶段三）
    goal_encoder = torch.load(config['goal_encoder_path'], map_location=device)
    goal_encoder.eval()
    for p in goal_encoder.parameters():
        p.requires_grad = False

    # 子目标编码器（阶段三）
    subgoal_encoder = torch.load(config['subgoal_encoder_path'], map_location=device)
    subgoal_encoder.eval()
    for p in subgoal_encoder.parameters():
        p.requires_grad = False

    # 动作解码器（阶段三）
    action_decoder = torch.load(config['action_decoder_path'], map_location=device)
    action_decoder.eval()
    for p in action_decoder.parameters():
        p.requires_grad = False

    # 低层策略（SAC actor, 阶段三）
    low_actor = torch.load(config['low_actor_path'], map_location=device)
    low_actor.eval()
    for p in low_actor.parameters():
        p.requires_grad = False

    # 高层策略（TD3 actor, 阶段三）
    high_actor = torch.load(config['high_actor_path'], map_location=device)
    high_actor.eval()
    for p in high_actor.parameters():
        p.requires_grad = False

    # Termination Critic（阶段三）
    termination_critic = torch.load(config['termination_critic_path'], map_location=device)
    termination_critic.eval()
    for p in termination_critic.parameters():
        p.requires_grad = False

    # GIA 网络（阶段三，只输出距离）
    gia = torch.load(config['gia_path'], map_location=device)
    gia.eval()
    for p in gia.parameters():
        p.requires_grad = False

    # -------------------- 环境初始化 --------------------
    env = config['env_creator']()   # 需实现 reset, step, get_goal_state, request_goal

    obs = env.reset()
    proprio, sensor, vision = parse_obs(obs)
    with torch.no_grad():
        z = state_encoder(proprio.unsqueeze(0).to(device),
                          sensor.unsqueeze(0).to(device),
                          vision.unsqueeze(0).to(device)).squeeze(0)

    initial_goal_state = env.get_goal_state()
    if isinstance(initial_goal_state, np.ndarray):
        initial_goal_state = torch.from_numpy(initial_goal_state).float()
    with torch.no_grad():
        z_goal = goal_encoder(initial_goal_state.unsqueeze(0).to(device)).squeeze(0)

    # 高层生成初始子目标
    with torch.no_grad():
        g = high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
        g_target = subgoal_encoder(g.unsqueeze(0)).squeeze(0)

    # -------------------- 推理参数 --------------------
    max_steps = config.get('max_inference_steps', 1000)
    term_threshold = config.get('term_threshold', 0.5)           # 终止概率阈值
    stagnation_window = config.get('stagnation_window', 10)      # GIA 停滞检测窗口长度
    duration_scale = config.get('duration_scale', 10)            # 动作持续时间缩放
    max_subgoal_steps = config.get('max_subgoal_steps', 50)      # 子目标最大步数

    # -------------------- GIA 滑动窗口 --------------------
    dist_window = deque(maxlen=stagnation_window)

    step = 0
    subgoal_step = 0
    done = False
    print("开始推理...")

    while not done and step < max_steps:
        # ----- 低层执行一步（单步动作）-----
        with torch.no_grad():
            # 低层策略输出动作分布，取均值（确定性）
            dist = low_actor(z.unsqueeze(0), g_target.unsqueeze(0))
            a_prim = dist.mean.squeeze(0)
            a_phys, duration_tensor = action_decoder(a_prim.unsqueeze(0))
            a_phys = a_phys.squeeze(0).cpu().numpy()
            duration = max(1, int(duration_tensor.item() * duration_scale))

        # 执行连续动作（duration 次）
        for _ in range(duration):
            obs, reward, done = env.step(a_phys)
            step += 1
            subgoal_step += 1
            if done or step >= max_steps:
                break

        # 编码新状态
        proprio, sensor, vision = parse_obs(obs)
        with torch.no_grad():
            z_next = state_encoder(proprio.unsqueeze(0).to(device),
                                   sensor.unsqueeze(0).to(device),
                                   vision.unsqueeze(0).to(device)).squeeze(0)

        # ----- Termination Critic 判断子目标是否终止 -----
        with torch.no_grad():
            term_prob = termination_critic(z_next.unsqueeze(0), g_target.unsqueeze(0)).item()

        # 如果子目标步数超限或终止概率超过阈值，则高层重新规划
        if subgoal_step >= max_subgoal_steps or np.random.random() < term_prob:
            print(f"步骤 {step}: 子目标终止 (步数: {subgoal_step}, 概率: {term_prob:.3f})")
            with torch.no_grad():
                g = high_actor(z_next.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                g_target = subgoal_encoder(g.unsqueeze(0)).squeeze(0)
            subgoal_step = 0

        # 更新当前状态
        z = z_next

        # ----- GIA 预测距离 + 停滞检测（最终目标更新）-----
        with torch.no_grad():
            # GIA 输入：(z, z_goal, g_target, uncertainty)
            uncertainty = torch.zeros(1, device=device)   # 可扩展为预测器方差
            pred_dist = gia(z.unsqueeze(0), z_goal.unsqueeze(0),
                            g_target.unsqueeze(0), uncertainty).item()

        dist_window.append(pred_dist)

        if len(dist_window) == stagnation_window:
            half = stagnation_window // 2
            early_avg = np.mean(list(dist_window)[:half])
            recent_avg = np.mean(list(dist_window)[half:])
            if recent_avg >= early_avg:   # 距离未下降，停滞
                print(f"步骤 {step}: GIA 检测到距离停滞 (早: {early_avg:.3f}, 近: {recent_avg:.3f})，请求新目标")
                new_goal_state = env.request_goal()
                if isinstance(new_goal_state, np.ndarray):
                    new_goal_state = torch.from_numpy(new_goal_state).float()
                with torch.no_grad():
                    z_goal = goal_encoder(new_goal_state.unsqueeze(0).to(device)).squeeze(0)
                # 目标改变后立即重新规划子目标
                with torch.no_grad():
                    g = high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                    g_target = subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                subgoal_step = 0
                dist_window.clear()

        # 打印进度
        if step % 50 == 0:
            remaining = torch.norm(z - z_goal).item()
            print(f"Step {step}, remaining distance: {remaining:.3f}, term prob: {term_prob:.3f}")

        if done:
            break

    print("推理结束。")
    env.close()


# ---------- 配置示例 ----------
if __name__ == "__main__":
    config = {
        'device': 'cuda',
        # 模型路径（阶段三保存的权重）
        'state_encoder_path': './checkpoints/stage2_state_encoder.pth',
        'predictor_path': './checkpoints/stage2_predictor.pth',
        'goal_encoder_path': './checkpoints/stage3_goal_encoder.pth',
        'subgoal_encoder_path': './checkpoints/stage3_subgoal_encoder.pth',
        'action_decoder_path': './checkpoints/stage3_action_decoder.pth',
        'low_actor_path': './checkpoints/stage3_low_actor.pth',
        'high_actor_path': './checkpoints/stage3_high_actor.pth',
        'termination_critic_path': './checkpoints/stage3_termination_critic.pth',
        'gia_path': './checkpoints/stage3_gia.pth',
        # 推理参数
        'term_threshold': 0.5,
        'stagnation_window': 10,
        'duration_scale': 10,
        'max_subgoal_steps': 50,
        'max_inference_steps': 1000,
        # 环境创建函数（需用户实现）
        'env_creator': lambda: UnityEnvWrapper(...),
    }
    inference_loop(config)