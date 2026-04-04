#!/usr/bin/env python3
# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""
GO2W SBX — SAC training with JAX backend.

Hyperparameters from QuadLoco Optuna Trial 72 (after_optuna.py):
  gamma=0.9842, lr=0.001440, batch=1024, train_freq=8,
  gradient_steps=256 (UTD=32), policy_delay=32, tau=0.01024,
  buffer=8M, layer_norm=True, optimizer=adamw

Usage (from IsaacLab directory):
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_sbx/train.py \\
        --num_envs 2048 --headless --timesteps 50000000
"""

import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
_SBX_DIR = os.path.dirname(os.path.abspath(__file__))

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

# JAX memory config — "false" prevents over-preallocation (QuadLoco after_optuna style)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GO2W SBX — SAC training (JAX backend).")
parser.add_argument("--num_envs",       type=int,  default=2048,        help="Number of parallel environments.")
parser.add_argument("--timesteps",      type=int,  default=50_000_000,  help="Total training timesteps.")
parser.add_argument("--seed",           type=int,  default=42,          help="Random seed (-1 = random).")
parser.add_argument("--checkpoint",     type=str,  default=None,        help="Resume from model .zip file.")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,  default=200)
parser.add_argument("--video_interval", type=int,  default=50_000)
parser.add_argument("--wandb",          action="store_true", default=False, help="Enable WandB logging.")
parser.add_argument("--wandb_project",  type=str,  default="go2w-sac",     help="WandB project name.")
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
from collections import deque

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
    import quadruped_lab   # noqa: F401
else:
    import robot_lab   # noqa: F401

if _LAB_MACHINE:
    from quadruped_lab.tasks.manager_based.locomotion.velocity.config.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg
else:
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg

LOG_ROOT = os.path.join(_SBX_DIR, "logs", "jax")

# ── Optuna Trial 72 hyperparameters ──────────────────────────────────────────
# Source: QuadLoco/scripts/sb3/after_optuna.py
# train_freq=8, gradient_steps=256 → UTD=32
# policy_delay=32: actor updated every 32 critic steps
# layer_norm=True: SBX-specific, improves stability
# optimizer=adamw: better regularization than adam
HYPERPARAMS = {
    "gamma":            0.984234743573426,
    "learning_rate":    0.001440190888183548,
    # Fixed (not auto): Go2W has 16 actions → SAC's default target=-16 drives auto ent_coef
    # to ~0.0009 by 45M steps, causing premature determinism. A1 (12 actions) collapses less.
    "ent_coef":         0.01,
    "batch_size":       1024,       # 2^10
    "train_freq":       8,          # 2^3
    "gradient_steps":   256,        # 2^8 → UTD=32
    "policy_delay":     32,         # 2^5
    "tau":              0.010237697694697378,
    "buffer_size":      8_000_000,
    "learning_starts":  100,
    "policy_kwargs": {
        "layer_norm":       True,
        "net_arch":         [512, 256, 128],
        "activation_fn":    jax.nn.elu,
        "optimizer_class":  optax.adamw,
    },
}


# ── Callbacks ─────────────────────────────────────────────────────────────────
class TimeoutCallback(BaseCallback):
    """Stop training after a fixed wall-clock duration (from after_optuna.py)."""

    def __init__(self, timeout: int):
        super().__init__()
        self.timeout = timeout
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return (time.time() - self.start_time) < self.timeout


class IsaacLabLoggerCallback(BaseCallback):
    """Logs Isaac Lab reward terms and completed episode stats to TensorBoard.

    Tracks COMPLETED episode rewards (not in-progress) to avoid the sawtooth
    artifact caused by synchronized resets across 2048 envs. Detects episode
    completions by watching for drops in _ep_rew_buf values.
    """

    def __init__(self, log_freq: int = 250, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._completed_rewards: deque = deque(maxlen=4096)
        self._completed_lengths: deque = deque(maxlen=4096)
        self._prev_ep_rew_buf = None
        self._prev_ep_len_buf = None

    def _on_training_start(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat
        for fmt in self.logger.output_formats:
            if isinstance(fmt, HumanOutputFormat):
                fmt.max_length = 100

    def _on_step(self) -> bool:
        # ── Detect completed episodes ─────────────────────────────────────────
        # When _ep_rew_buf[i] drops sharply, env i just reset → capture the
        # previous value as the completed episode reward.
        try:
            curr_rew = self.training_env._ep_rew_buf.copy()
            curr_len = self.training_env._ep_len_buf.copy()
            if self._prev_ep_rew_buf is not None:
                # Episode ended: current buf reset to ~0 but previous was > 0
                completed = (curr_len < self._prev_ep_len_buf * 0.5) & (self._prev_ep_len_buf > 10)
                if completed.any():
                    for r in self._prev_ep_rew_buf[completed].tolist():
                        self._completed_rewards.append(r)
                    for l in self._prev_ep_len_buf[completed].tolist():
                        self._completed_lengths.append(l)
            self._prev_ep_rew_buf = curr_rew
            self._prev_ep_len_buf = curr_len
        except Exception:
            pass

        if self.n_calls % self.log_freq != 0:
            return True

        # ── Log completed episode stats ───────────────────────────────────────
        if self._completed_rewards:
            self.logger.record("rollout/ep_rew_mean", float(np.mean(self._completed_rewards)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(self._completed_lengths)))

        # ── Log Isaac Lab per-term reward breakdown ───────────────────────────
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

        self.logger.dump(step=self.num_timesteps)
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

    # Clip action space to [-2, 2] — env applies scale=0.5 → actual joint offset [-1.0, 1.0] rad
    # [-3, 3] caused joint_pos_penalty=-0.75 (robot propping joints at extremes)
    # [-1, 1] was too small (only [-0.5, 0.5] rad, limited locomotion)
    env.action_space = gym.spaces.Box(
        low=-2.0, high=2.0, shape=env.action_space.shape, dtype=np.float32
    )

    print(f"[INFO] Obs space : {env.observation_space}")
    print(f"[INFO] Act space : {env.action_space}")

    # ── Agent ─────────────────────────────────────────────────────────────────
    hyperparams = {**HYPERPARAMS, "seed": seed}

    if args_cli.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.checkpoint}")
        model = sbx.SAC.load(args_cli.checkpoint, env=env)
    else:
        model = sbx.SAC("MlpPolicy", env, tensorboard_log=log_dir, verbose=1, **hyperparams)

    # ── Callbacks ─────────────────────────────────────────────────────────────
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
                name=f"go2w_sbx_{timestamp}",
                config={**hyperparams, "num_envs": args_cli.num_envs, "task": TASK_NAME},
                sync_tensorboard=True,
                reinit=True,
                settings=wandb.Settings(console="off"),
            )
            callbacks.append(WandbCallback(verbose=1))
            print(f"[INFO] WandB project     : {args_cli.wandb_project}")

    # ── Training summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GO2W SBX — SAC Training (JAX backend, Optuna Trial 72)")
    print("=" * 60)
    print(f"  Task              : {TASK_NAME}")
    print(f"  Envs              : {args_cli.num_envs}")
    print(f"  Timesteps         : {args_cli.timesteps:,}")
    print(f"  LR                : {hyperparams['learning_rate']}")
    print(f"  Gamma / tau       : {hyperparams['gamma']:.4f} / {hyperparams['tau']:.5f}")
    print(f"  Batch size        : {hyperparams['batch_size']}")
    print(f"  train_freq        : {hyperparams['train_freq']}  →  UTD = {hyperparams['gradient_steps'] / hyperparams['train_freq']:.0f}")
    print(f"  policy_delay      : {hyperparams['policy_delay']}")
    print(f"  buffer_size       : {hyperparams['buffer_size'] / 1e6:.0f}M")
    print(f"  layer_norm        : True  |  optimizer: adamw")
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
            log_interval=100,
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
