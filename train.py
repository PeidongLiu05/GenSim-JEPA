"""
GenSim-JEPA 三阶段训练入口脚本。

用法:
    python train.py --config configs/train/models/gsjepa.yaml --env_path /path/to/unity.x86_64
"""

import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

from envs.unity_env_wrapper import UnityEnvWrapper
from buffers.hierarchical_buffer import HierarchicalReplayBuffer
from networks.state_encoder import StateEncoder
from networks.action_encoder import ActionEncoder
from networks.action_decoder import ActionDecoder
from networks.goal_encoder import GoalEncoder
from networks.subgoal_encoder import SubgoalEncoder
from networks.low_level_policy import LowLevelPolicy
from networks.low_level_critic import LowLevelCritic
from networks.high_level_policy import HighLevelPolicy
from networks.high_level_critic import HighLevelCritic
from networks.termination_critic import TerminationCritic
from networks.goal_info_acquirer import GoalInfoAcquirer
from networks.lewm_predictor import LeWMPredictor


class Config:
    """支持 cfg.attr 和 cfg.get('key', default) 两种访问方式的配置类。"""
    def __init__(self, d):
        self._dict = d
        self.__dict__.update(d)
    def __getitem__(self, k):
        return self._dict[k]
    def __setitem__(self, k, v):
        self._dict[k] = v
        setattr(self, k, v)
    def get(self, k, default=None):
        return self._dict.get(k, default)


def build_networks(cfg):
    """根据配置构建所有网络。"""
    device = torch.device(cfg['device'])

    state_encoder = StateEncoder(
        proprio_dim=cfg['proprio_dim'],
        sensor_dim=cfg['sensor_dim'],
        vision_c=cfg['vision_c'],
        vision_h=cfg['vision_h'],
        vision_w=cfg['vision_w'],
        embed_dim=cfg['latent_dim'],
    ).to(device)

    action_encoder = ActionEncoder(
        proprio_dim=cfg['proprio_dim'], action_dim=cfg['action_dim'], prim_dim=cfg['prim_dim'],
    ).to(device)

    action_decoder = ActionDecoder(
        proprio_dim=cfg['proprio_dim'], prim_dim=cfg['prim_dim'], action_dim=cfg['action_dim'],
    ).to(device)

    goal_encoder = GoalEncoder(
        state_dim=cfg['proprio_dim'], embed_dim=cfg['goal_dim'],
    ).to(device)

    subgoal_encoder = SubgoalEncoder(
        g_dim=cfg['subgoal_dim'], z_dim=cfg['latent_dim'],
    ).to(device)

    low_actor = LowLevelPolicy(
        state_dim=cfg['latent_dim'], goal_dim=cfg['latent_dim'], action_dim=cfg['prim_dim'],
    ).to(device)

    low_critic_1 = LowLevelCritic(
        state_dim=cfg['latent_dim'], goal_dim=cfg['latent_dim'], action_dim=cfg['action_dim'],
    ).to(device)
    low_critic_2 = LowLevelCritic(
        state_dim=cfg['latent_dim'], goal_dim=cfg['latent_dim'], action_dim=cfg['action_dim'],
    ).to(device)

    use_high = cfg.get('use_hierarchy', True)
    if use_high:
        high_actor = HighLevelPolicy(
            state_dim=cfg['latent_dim'], goal_dim=cfg['latent_dim'], subgoal_dim=cfg['subgoal_dim'],
        ).to(device)
        high_critic_1 = HighLevelCritic(
            state_dim=cfg['latent_dim'], goal_dim=cfg['latent_dim'], subgoal_dim=cfg['subgoal_dim'],
        ).to(device)
        high_critic_2 = HighLevelCritic(
            state_dim=cfg['latent_dim'], goal_dim=cfg['latent_dim'], subgoal_dim=cfg['subgoal_dim'],
        ).to(device)
        termination_critic = TerminationCritic(
            state_dim=cfg['latent_dim'], subgoal_dim=cfg['latent_dim'],
        ).to(device)
        gia = GoalInfoAcquirer(
            state_dim=cfg['latent_dim'], goal_dim=cfg['goal_dim'], subgoal_dim=cfg['latent_dim'],
        ).to(device)
    else:
        high_actor = high_critic_1 = high_critic_2 = termination_critic = gia = None

    predictor = LeWMPredictor(
        latent_dim=cfg['latent_dim'], action_dim=cfg['prim_dim'],
    ).to(device)

    nets = {
        'state_encoder': state_encoder,
        'action_encoder': action_encoder,
        'action_decoder': action_decoder,
        'goal_encoder': goal_encoder,
        'subgoal_encoder': subgoal_encoder,
        'low_actor': low_actor,
        'low_critic_1': low_critic_1,
        'low_critic_2': low_critic_2,
        'high_actor': high_actor,
        'high_critic_1': high_critic_1,
        'high_critic_2': high_critic_2,
        'termination_critic': termination_critic,
        'gia': gia,
        'predictor': predictor,
    }
    # 非分层模式：释放未使用的网络节省显存
    if not use_high:
        for k in ['high_actor', 'high_critic_1', 'high_critic_2', 'termination_critic', 'gia',
                   'goal_encoder', 'subgoal_encoder']:
            nets[k] = None
    return nets


def check_dimensions(cfg):
    assert cfg['proprio_dim'] >= 10
    assert cfg['sensor_dim'] >= 4
    assert cfg['action_dim'] > 0
    assert cfg['prim_dim'] > 0
    assert cfg['latent_dim'] == cfg['goal_dim']
    print(f"[维度] proprio={cfg['proprio_dim']} sensor={cfg['sensor_dim']} "
          f"action={cfg['action_dim']} prim={cfg['prim_dim']} latent={cfg['latent_dim']}")


def train_stage1(cfg, env, nets, buffer, resume=None, resume_ep=0):
    from trainers.stage1_trainer import Stage1Trainer
    target_entropy = -cfg['target_entropy']
    use_high = nets['high_actor'] is not None
    trainer = Stage1Trainer(
        config=cfg, env=env, target_entropy=target_entropy,
        state_encoder=nets['state_encoder'],
        action_encoder=nets['action_encoder'],
        goal_encoder=nets['goal_encoder'],
        subgoal_encoder=nets['subgoal_encoder'],
        action_decoder=nets['action_decoder'],
        low_actor=nets['low_actor'],
        low_critic_1=nets['low_critic_1'],
        low_critic_2=nets['low_critic_2'],
        target_low_critic_1=deepcopy(nets['low_critic_1']),
        target_low_critic_2=deepcopy(nets['low_critic_2']),
        high_actor=nets['high_actor'],
        target_high_actor=deepcopy(nets['high_actor']) if use_high else None,
        high_critic_1=nets['high_critic_1'],
        high_critic_2=nets['high_critic_2'],
        target_high_critic_1=deepcopy(nets['high_critic_1']) if use_high else None,
        target_high_critic_2=deepcopy(nets['high_critic_2']) if use_high else None,
        termination_critic=nets['termination_critic'],
        gia_network=nets['gia'],
        replay_buffer=buffer,
    )
    if resume:
        start_ep = trainer.load_checkpoint(resume, resume_ep)
    else:
        start_ep = 0
    trainer.train(cfg['stage1_episodes'], start_ep=start_ep)


def train_stage2(cfg, nets, buffer, resume=None):
    from trainers.stage2_trainer import Stage2Trainer
    trainer = Stage2Trainer(
        config=cfg,
        state_encoder=nets['state_encoder'],
        action_encoder=nets['action_encoder'],
        predictor=nets['predictor'],
        lewm_buffer=buffer,
    )
    if resume:
        trainer.load_checkpoint(resume)
    trainer.train(cfg['stage2_epochs'])


def train_stage3(cfg, env, nets, buffer, resume=None, resume_ep=0):
    from trainers.stage3_trainer import Stage3Trainer
    trainer = Stage3Trainer(
        config=cfg, env=env,
        state_encoder=nets['state_encoder'],
        lewm_predictor=nets['predictor'],
        goal_encoder=nets['goal_encoder'],
        subgoal_encoder=nets['subgoal_encoder'],
        action_decoder=nets['action_decoder'],
        low_actor=nets['low_actor'],
        low_critic_1=nets['low_critic_1'],
        low_critic_2=nets['low_critic_2'],
        high_actor=nets['high_actor'],
        high_critic_1=nets['high_critic_1'],
        high_critic_2=nets['high_critic_2'],
        termination_critic=nets['termination_critic'],
        gia_network=nets['gia'],
        replay_buffer=buffer,
    )
    if resume:
        start_ep = trainer.load_checkpoint(resume, resume_ep)
    else:
        start_ep = 0
    trainer.train(cfg['stage3_episodes'], start_ep=start_ep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/models/gsjepa.yaml')
    parser.add_argument('--env_path', type=str, default=None)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--resume', type=str, default=None,
                        help='续接训练用的 checkpoint 路径，如 ./checkpoints/stage1_nohier_ep10000.pt')
    parser.add_argument('--resume_ep', type=int, default=0,
                        help='续接时的起始 episode 编号（覆盖 checkpoint 中的 ep）')
    args = parser.parse_args()
    env_path = None if args.env_path == 'None' else args.env_path

    with open(args.config, 'r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)

    cfg = Config(raw_cfg)
    check_dimensions(cfg)

    nets = build_networks(cfg)
    buffer = HierarchicalReplayBuffer()

    timeout = 300 if env_path is None else 60
    print(f"[train] 连接 Unity {'Editor' if env_path is None else env_path}，超时 {timeout}s，请确保 Unity Editor 已打开场景并按下 Play…")
    env = UnityEnvWrapper(
        env_path=env_path,
        worker_id=0,
        proprio_dim=cfg['proprio_dim'],
        sensor_dim=cfg['sensor_dim'],
        camera_height=cfg['vision_h'],
        camera_width=cfg['vision_w'],
        timeout_wait=timeout,
    )

    try:
        if args.stage == 1:
            train_stage1(cfg, env, nets, buffer, resume=args.resume, resume_ep=args.resume_ep)
        elif args.stage == 2:
            train_stage2(cfg, nets, buffer, resume=args.resume)
        else:
            train_stage3(cfg, env, nets, buffer, resume=args.resume, resume_ep=args.resume_ep)
    finally:
        env.close()


if __name__ == '__main__':
    main()
