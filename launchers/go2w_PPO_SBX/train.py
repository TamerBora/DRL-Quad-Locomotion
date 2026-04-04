#!/usr/bin/env python3
# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""
GO2W SBX — PPO training with JAX backend.

Ported from SB3 (PyTorch) to SBX (JAX) for a valid benchmark against SBX SAC.
Network architecture and PPO hyperparameters unchanged; only backend switched.

Usage (from IsaacLab directory):
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_PPO_SBX/train.py \\
        --num_envs 2048 --headless --timesteps 50000000
"""

import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")

if _LAB_MACHINE:
    _ROBOTICS_DIR = "/home/roblab/quadruped_lab"
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "source", "quadruped_lab"))
    TASK_NAME = "QuadrupedLab-Isaac-Velocity-Flat-Unitree-Go2W-v0"
else:
    _ROBOTICS_DIR = os.path.expanduser("~/robotics")
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "launchers", "go2w_v3", "source"))
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "IsaacLab", "source", "isaaclab"))
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "IsaacLab", "source", "isaaclab_tasks"))
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "IsaacLab", "source", "isaaclab_rl"))
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "robot_lab", "source", "robot_lab"))
    TASK_NAME = "RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0"

# JAX memory config — prevents over-preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GO2W SBX — PPO training (JAX backend).")
parser.add_argument("--num_envs",       type=int,  default=2048,        help="Number of parallel environments.")
parser.add_argument("--timesteps",      type=int,  default=50_000_000,  help="Total training timesteps.")
parser.add_argument("--seed",           type=int,  default=42,          help="Random seed (-1 = random).")
parser.add_argument("--checkpoint",     type=str,  default=None,        help="Resume from model .zip file.")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,  default=200)
parser.add_argument("--video_interval", type=int,  default=50_000)
parser.add_argument("--wandb",          action="store_true", default=False, help="Enable WandB logging.")
parser.add_argument("--wandb_project",  type=str,  default="go2w-ppo",     help="WandB project name.")
parser.add_argument("--timeout",        type=int,  default=0,              help="Training timeout in seconds (0 = no timeout).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import logging
import random
import time
from datetime import datetime

import gymnasium as gym
import jax.nn
import numpy as np
import optax
import sbx

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

if _LAB_MACHINE:
    import quadruped_lab  # noqa: F401
    from quadruped_lab.tasks.manager_based.locomotion.velocity.config.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg
else:
    import robot_lab  # noqa: F401
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg

LOG_ROOT = os.path.join(_DIR, "logs", "ppo_jax")


# ── Callbacks ─────────────────────────────────────────────────────────────────
class TimeoutCallback(BaseCallback):
    """Stop training after a fixed wall-clock duration."""

    def __init__(self, timeout: int):
        super().__init__()
        self.timeout = timeout
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return (time.time() - self.start_time) < self.timeout


class IsaacLabLoggerCallback(BaseCallback):
    """Logs Isaac Lab per-term reward breakdown to TensorBoard."""

    def _on_training_start(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat
        for fmt in self.logger.output_formats:
            if isinstance(fmt, HumanOutputFormat):
                fmt.max_length = 100

    def _on_step(self) -> bool:
        try:
            isaac_env = self.training_env.unwrapped.unwrapped
            if "log" in isaac_env.extras:
                for key, value in isaac_env.extras["log"].items():
                    try:
                        self.logger.record(
                            f"isaac/{key}",
                            float(value.mean()) if hasattr(value, "mean") else float(value)
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        return True


def main():
    seed = args_cli.seed if args_cli.seed >= 0 else random.randint(0, 10_000)

    # ── Env config ────────────────────────────────────────────────────────────
    env_cfg = UnitreeGo2WFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

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

    env = Sb3VecEnvWrapper(env, fast_variant=True)

    print(f"[INFO] Obs space : {env.observation_space}")
    print(f"[INFO] Act space : {env.action_space}")

    # ── Agent ─────────────────────────────────────────────────────────────────
    # n_steps=24: rollout length per env (matches Isaac Lab PPO default).
    # batch_size = n_steps * n_envs / 4 minibatches per rollout.
    # layer_norm not available in SBX PPO (SAC-specific feature).
    # log_std_init=-2.0 → initial std=exp(-2)≈0.135 (vs default 0 → std=1.0).
    # Low initial std keeps action_rate_l2 penalty small while the value function
    # bootstraps, preventing the KL/clip_fraction explosion seen with high std.
    policy_kwargs = dict(
        net_arch=[512, 256, 128],
        activation_fn=jax.nn.elu,
        optimizer_class=optax.adam,
        log_std_init=-2.0,
    )

    n_steps    = 32
    batch_size = n_steps * args_cli.num_envs // 4  # 4 minibatches per rollout

    if args_cli.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.checkpoint}")
        model = sbx.PPO.load(args_cli.checkpoint, env=env)
    else:
        model = sbx.PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=1.0,
            max_grad_norm=1.0,
            target_kl=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            seed=seed,
            verbose=1,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # Save every 5M steps (10 checkpoints over 50M), keep only the last 3.
    # Previously saved every 100K → ~500 checkpoints → ~2.5GB per run.
    ckpt_freq = max(5_000_000 // args_cli.num_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    callbacks = [checkpoint_callback, IsaacLabLoggerCallback()]

    if args_cli.timeout > 0:
        callbacks.append(TimeoutCallback(args_cli.timeout))
        print(f"[INFO] Training timeout  : {args_cli.timeout}s")

    if args_cli.wandb:
        if not _WANDB_AVAILABLE:
            print("[WARN] WandB not installed — skipping (pip install wandb).")
        else:
            wandb.init(
                project=args_cli.wandb_project,
                name=f"go2w_ppo_{timestamp}",
                config={
                    "num_envs": args_cli.num_envs,
                    "task": TASK_NAME,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "learning_rate": 1e-3,
                    "gamma": 0.99,
                    "ent_coef": 0.008,
                },
                sync_tensorboard=True,
                reinit=True,
                settings=wandb.Settings(console="off"),
            )
            callbacks.append(WandbCallback(verbose=1))
            print(f"[INFO] WandB project     : {args_cli.wandb_project}")

    # ── Training summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GO2W SBX — PPO Training (JAX backend)")
    print("=" * 60)
    print(f"  Task              : {TASK_NAME}")
    print(f"  Envs              : {args_cli.num_envs}")
    print(f"  Timesteps         : {args_cli.timesteps:,}")
    print(f"  n_steps           : {n_steps}  →  rollout = {n_steps * args_cli.num_envs:,}")
    print(f"  batch_size        : {batch_size}  (4 minibatches)")
    print(f"  n_epochs          : 5")
    print(f"  gamma / gae_lambda: 0.99 / 0.95")
    print(f"  ent_coef          : 0.008")
    print(f"  Network arch      : [512, 256, 128] ELU  |  optimizer: adam")
    print(f"  Seed              : {seed}")
    print(f"  Log directory     : {log_dir}")
    print("=" * 60 + "\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=args_cli.timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10,
            reset_num_timesteps=args_cli.checkpoint is None,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = os.path.join(log_dir, "model")
    model.save(model_path)
    print(f"[INFO] Model saved   : {model_path}.zip")
    print(f"[INFO] Training time : {time.time() - start_time:.1f}s")

    if args_cli.wandb and _WANDB_AVAILABLE:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
