#!/usr/bin/env python3
# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""
GO2W — SB3 PPO training on rough terrain (stairs, obstacles, slopes).

Usage (from IsaacLab directory):
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_PPO_SB3/train_rough.py \
        --num_envs 4096 --headless --timesteps 100000000

    # Resume from checkpoint
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_PPO_SB3/train_rough.py \
        --num_envs 4096 --headless --checkpoint /path/to/model.zip
"""

import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOTICS_DIR = os.path.expanduser("~/robotics")

sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "launchers", "go2w_v3", "source"))
sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "IsaacLab", "source", "isaaclab"))
sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "IsaacLab", "source", "isaaclab_tasks"))
sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "IsaacLab", "source", "isaaclab_rl"))
sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "robot_lab", "source", "robot_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GO2W SB3 PPO rough terrain training.")
parser.add_argument("--num_envs",   type=int,  default=4096,        help="Number of parallel environments.")
parser.add_argument("--timesteps",  type=int,  default=100_000_000, help="Total training timesteps.")
parser.add_argument("--seed",       type=int,  default=42,          help="Random seed.")
parser.add_argument("--checkpoint", type=str,  default=None,        help="Resume from model .zip file.")
parser.add_argument("--video",      action="store_true", default=False)
parser.add_argument("--video_length",   type=int, default=200)
parser.add_argument("--video_interval", type=int, default=50_000)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import random
import time
from datetime import datetime

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import robot_lab      # noqa: F401
import quadruped_lab  # noqa: F401

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import UnitreeGo2WRoughEnvCfg

TASK_NAME = "RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0"
LOG_ROOT  = os.path.join(_DIR, "logs", "ppo_rough")


class IsaacLabLoggerCallback(BaseCallback):
    """Reads Isaac Lab env.extras['log'] each step and writes to TensorBoard."""

    def _on_step(self) -> bool:
        try:
            isaac_env = self.training_env.unwrapped.unwrapped
            if "log" in isaac_env.extras:
                for key, value in isaac_env.extras["log"].items():
                    try:
                        self.logger.record(f"isaac/{key}", float(value.mean()))
                    except Exception:
                        pass
        except Exception:
            pass
        return True


def main():
    seed = args_cli.seed if args_cli.seed >= 0 else random.randint(0, 10_000)

    env_cfg = UnitreeGo2WRoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    device = env_cfg.sim.device

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir   = os.path.join(LOG_ROOT, timestamp)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    # ── Environment ───────────────────────────────────────────────────────────
    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env)

    print(f"[INFO] Obs space : {env.observation_space}")
    print(f"[INFO] Act space : {env.action_space}")

    # ── Agent ─────────────────────────────────────────────────────────────────
    # Larger network than flat terrain to process 187-dim height scan.
    # n_steps=24 matches Isaac Lab PPO default rollout length.
    # Higher timesteps needed — rough terrain curriculum takes longer to climb.
    policy_kwargs = dict(
        net_arch=[512, 256, 128],
        activation_fn=nn.ELU,
    )

    if args_cli.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.checkpoint}")
        agent = PPO.load(args_cli.checkpoint, env=env, device=device)
    else:
        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=24,
            batch_size=24 * args_cli.num_envs // 4,  # 4 minibatches per rollout
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.008,
            vf_coef=1.0,
            max_grad_norm=1.0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            seed=seed,
            device=device,
            verbose=1,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_freq = max(100_000 // args_cli.num_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="model",
        save_vecnormalize=False,
        verbose=1,
    )

    # ── Training summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GO2W — SB3 PPO Rough Terrain Training")
    print("=" * 60)
    print(f"  Task         : {TASK_NAME}")
    print(f"  Terrain      : stairs, obstacles, rough, slopes (curriculum)")
    print(f"  Obs space    : ~234 dims (57 base + 187 height scan)")
    print(f"  Envs         : {args_cli.num_envs}")
    print(f"  Timesteps    : {args_cli.timesteps:,}")
    print(f"  n_steps      : {agent.n_steps}")
    print(f"  batch_size   : {agent.batch_size}")
    print(f"  n_epochs     : {agent.n_epochs}")
    print(f"  Network arch : [512, 256, 128] ELU")
    print(f"  Seed         : {seed}")
    print(f"  Device       : {device}")
    print(f"  Log dir      : {log_dir}")
    print("=" * 60 + "\n")

    start_time = time.time()
    try:
        agent.learn(
            total_timesteps=args_cli.timesteps,
            callback=[checkpoint_callback, IsaacLabLoggerCallback()],
            progress_bar=True,
            log_interval=1,
            reset_num_timesteps=args_cli.checkpoint is None,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    model_path = os.path.join(log_dir, "model")
    agent.save(model_path)
    print(f"[INFO] Model saved   : {model_path}.zip")
    print(f"[INFO] Training time : {time.time() - start_time:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
