"""
推理脚本：连接 Unity Editor，加载训练好的低层策略，驱动人形机器人行走。
支持分层 (use_hierarchy=true) 和非分层两种模式。
"""

import torch
import numpy as np
from collections import deque
from pathlib import Path
from envs.unity_env_wrapper import UnityEnvWrapper
import yaml

from networks.state_encoder import StateEncoder
from networks.low_level_policy import LowLevelPolicy
from networks.action_decoder import ActionDecoder
from networks.lewm_predictor import LeWMPredictor
from networks.goal_encoder import GoalEncoder
from networks.subgoal_encoder import SubgoalEncoder
from networks.high_level_policy import HighLevelPolicy
from networks.termination_critic import TerminationCritic
from networks.goal_info_acquirer import GoalInfoAcquirer


def parse_obs(obs):
    proprio = torch.from_numpy(obs['proprio']).float()
    sensor = torch.from_numpy(obs['sensor']).float()
    vision = torch.from_numpy(obs['vision']).float()
    return proprio, sensor, vision


def _load_model(model_class, path, device, ckpt_dict=None, ckpt_key=None, **kwargs):
    model = model_class(**kwargs).to(device)
    if ckpt_dict is not None and ckpt_key is not None:
        if ckpt_key not in ckpt_dict:
            print(f"  [skip] {ckpt_key} not found in checkpoint")
            return None
        state_dict = ckpt_dict[ckpt_key]
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {ckpt_key or path}: missing keys: {missing}")
    if unexpected:
        print(f"  [warn] {ckpt_key or path}: unexpected keys: {unexpected}")
    model.eval()
    return model


def inference_loop(config):
    device = torch.device(config.get('device', 'cuda'))
    use_hierarchy = config.get('use_hierarchy', False)
    ckpt_dict = config.get('_ckpt_dict', None)

    def _load_from_ckpt(Cls, key, **kwargs):
        return _load_model(Cls, config[key + '_path'], device,
                           ckpt_dict=ckpt_dict, ckpt_key=key, **kwargs)

    # ---------- 加载必要模型 ----------
    state_encoder = _load_from_ckpt(StateEncoder, 'state_encoder',
                                    proprio_dim=config['proprio_dim'],
                                    sensor_dim=config['sensor_dim'],
                                    embed_dim=config['latent_dim'],
                                    vision_c=config['vision_c'],
                                    vision_h=config['vision_h'],
                                    vision_w=config['vision_w'])

    low_actor = _load_from_ckpt(LowLevelPolicy, 'low_actor',
                                state_dim=config['latent_dim'],
                                goal_dim=config['latent_dim'],
                                action_dim=config['prim_dim'])

    action_decoder = _load_from_ckpt(ActionDecoder, 'action_decoder',
                                     prim_dim=config['prim_dim'],
                                     action_dim=config['action_dim'])

    if use_hierarchy:
        lewm_predictor = _load_from_ckpt(LeWMPredictor, 'predictor',
                                         latent_dim=config['latent_dim'],
                                         action_dim=config['prim_dim'])
        goal_encoder = _load_from_ckpt(GoalEncoder, 'goal_encoder',
                                       state_dim=config['proprio_dim'],
                                       embed_dim=config['goal_dim'])
        subgoal_encoder = _load_from_ckpt(SubgoalEncoder, 'subgoal_encoder',
                                          g_dim=config['subgoal_dim'],
                                          z_dim=config['latent_dim'])
        high_actor = _load_from_ckpt(HighLevelPolicy, 'high_actor',
                                     state_dim=config['latent_dim'],
                                     goal_dim=config['latent_dim'],
                                     subgoal_dim=config['subgoal_dim'])
        termination_critic = _load_from_ckpt(TerminationCritic, 'termination_critic',
                                             state_dim=config['latent_dim'],
                                             subgoal_dim=config['latent_dim'])
        gia = _load_from_ckpt(GoalInfoAcquirer, 'gia',
                              state_dim=config['latent_dim'],
                              goal_dim=config['goal_dim'],
                              subgoal_dim=config['latent_dim'])

    # ---------- 环境 ----------
    env = UnityEnvWrapper(
        env_path=config.get('env_path', None),
        worker_id=config.get('worker_id', 0),
        proprio_dim=config.get('proprio_dim', 34),
        sensor_dim=config.get('sensor_dim', 38),
        camera_height=config.get('vision_h', 64),
        camera_width=config.get('vision_w', 64),
        no_graphics=False,
    )

    obs = env.reset()
    proprio, sensor, vision = parse_obs(obs)

    with torch.no_grad():
        z = state_encoder(proprio.to(device),
                          sensor.to(device),
                          vision.to(device)).squeeze(0)

    # ---- 非分层模式：低层策略直接控制 ----
    if not use_hierarchy:
        max_steps = config.get('max_inference_steps', 5000)
        print("推理开始（非分层模式 | 低层 SAC 策略），Ctrl+C 终止...")
        episode = 0
        try:
            while True:
                g_target = torch.zeros_like(z)
                step = 0
                done = False
                while not done and step < max_steps:
                    with torch.no_grad():
                        a_prim, _ = low_actor(z.unsqueeze(0), g_target.unsqueeze(0), deterministic=True)
                        a_phys = action_decoder(proprio.to(device), a_prim)
                        a_phys = a_phys.squeeze(0).cpu().numpy()
                    obs, reward, done = env.step(a_phys)
                    step += 1
                    if done or step >= max_steps:
                        break
                    proprio, sensor, vision = parse_obs(obs)
                    with torch.no_grad():
                        z = state_encoder(proprio.to(device),
                                          sensor.to(device),
                                          vision.to(device)).squeeze(0)
                episode += 1
                print(f"Episode {episode} 结束，共 {step} 步，reward={reward[0]:.2f}")
                # 自动重置下一轮
                obs = env.reset()
                proprio, sensor, vision = parse_obs(obs)
                with torch.no_grad():
                    z = state_encoder(proprio.to(device),
                                      sensor.to(device),
                                      vision.to(device)).squeeze(0)
        except KeyboardInterrupt:
            print("\n推理终止。")
        finally:
            env.close()
        return

    # ---- 分层模式（原逻辑） ----
    initial_goal_state = env.get_goal_state()
    initial_goal_state = torch.from_numpy(initial_goal_state).float()
    with torch.no_grad():
        z_goal = goal_encoder(initial_goal_state.unsqueeze(0).to(device)).squeeze(0)

    with torch.no_grad():
        g = high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
        g_target = subgoal_encoder(g.unsqueeze(0)).squeeze(0)

    max_steps = config.get('max_inference_steps', 1000)
    term_threshold = config.get('term_threshold', 0.5)
    stagnation_window = config.get('stagnation_window', 10)
    max_subgoal_steps = config.get('max_subgoal_steps', 50)
    dist_window = deque(maxlen=stagnation_window)

    step = 0
    subgoal_step = 0
    done = False
    print("推理开始（分层模式）...")

    while not done and step < max_steps:
        with torch.no_grad():
            a_prim, _ = low_actor(z.unsqueeze(0), g_target.unsqueeze(0), deterministic=True)
            a_phys = action_decoder(proprio.to(device), a_prim)
            a_phys = a_phys.squeeze(0).cpu().numpy()

        obs, reward, done = env.step(a_phys)
        step += 1
        subgoal_step += 1

        proprio, sensor, vision = parse_obs(obs)
        with torch.no_grad():
            z_next = state_encoder(proprio.to(device),
                                   sensor.to(device),
                                   vision.to(device)).squeeze(0)

        with torch.no_grad():
            term_prob = termination_critic(z_next.unsqueeze(0), g_target.unsqueeze(0)).item()

        if subgoal_step >= max_subgoal_steps or np.random.random() < term_prob:
            print(f"Step {step}: subgoal terminated (steps={subgoal_step}, prob={term_prob:.3f})")
            with torch.no_grad():
                g = high_actor(z_next.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                g_target = subgoal_encoder(g.unsqueeze(0)).squeeze(0)
            subgoal_step = 0

        z = z_next

        with torch.no_grad():
            uncertainty = torch.zeros(1, device=device)
            pred_dist = gia(z.unsqueeze(0), z_goal.unsqueeze(0),
                            g_target.unsqueeze(0), uncertainty).item()
        dist_window.append(pred_dist)

        if len(dist_window) == stagnation_window:
            half = stagnation_window // 2
            early_avg = np.mean(list(dist_window)[:half])
            recent_avg = np.mean(list(dist_window)[half:])
            if recent_avg >= early_avg:
                print(f"Step {step}: GIA stagnation detected, requesting new goal")
                new_goal_state = env.request_goal()
                new_goal_state = torch.from_numpy(new_goal_state).float()
                with torch.no_grad():
                    z_goal = goal_encoder(new_goal_state.unsqueeze(0).to(device)).squeeze(0)
                    g = high_actor(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
                    g_target = subgoal_encoder(g.unsqueeze(0)).squeeze(0)
                subgoal_step = 0
                dist_window.clear()

        if step % 50 == 0:
            remaining = torch.norm(z - z_goal).item()
            print(f"Step {step}, remaining: {remaining:.3f}, term prob: {term_prob:.3f}")

    print(f"推理结束。共 {step} 步。")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/models/gsjepa.yaml')
    parser.add_argument('--env_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg['env_path'] = None if args.env_path == 'None' else args.env_path
    cfg['device'] = args.device
    ckpt_path = Path(args.checkpoint_dir).resolve()
    is_single_ckpt = ckpt_path.suffix in ('.pt', '.pth')

    use_h = cfg.get('use_hierarchy', True)

    if is_single_ckpt:
        # 从单个 stage1 checkpoint .pt 文件加载
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        print(f"从单文件加载 checkpoint: {ckpt_path}")
        cfg.update({
            '_ckpt_dict': ckpt,
            'state_encoder_path': ckpt_path,
            'low_actor_path': ckpt_path,
            'action_decoder_path': ckpt_path,
            'predictor_path': ckpt_path,
            'goal_encoder_path': ckpt_path,
            'subgoal_encoder_path': ckpt_path,
            'high_actor_path': ckpt_path,
            'termination_critic_path': ckpt_path,
            'gia_path': ckpt_path,
        })
    else:
        cfg['state_encoder_path'] = str(ckpt_path / 'stage2_state_encoder.pth')
        cfg['predictor_path'] = str(ckpt_path / 'stage2_predictor.pth')
        cfg['goal_encoder_path'] = str(ckpt_path / 'stage3_goal_encoder.pth')
        cfg['subgoal_encoder_path'] = str(ckpt_path / 'stage3_subgoal_encoder.pth')
        cfg['action_decoder_path'] = str(ckpt_path / 'stage3_action_decoder.pth')
        cfg['low_actor_path'] = str(ckpt_path / 'stage3_low_actor.pth')
        if use_h:
            cfg['high_actor_path'] = str(ckpt_path / 'stage3_high_actor.pth')
            cfg['termination_critic_path'] = str(ckpt_path / 'stage3_termination_critic.pth')
            cfg['gia_path'] = str(ckpt_path / 'stage3_gia.pth')

    inference_loop(cfg)
