#!/usr/bin/env python3
# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""
GO2W SBX — PPO evaluation.

Usage (from IsaacLab directory):
    # Auto-select latest checkpoint
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_PPO_SBX/play.py

    # Explicit checkpoint
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_PPO_SBX/play.py \\
        --checkpoint /home/tamer/robotics/launchers/go2w_PPO_SBX/logs/ppo_jax/<run>/model.zip

    # With video
    ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_PPO_SBX/play.py --video
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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GO2W SBX PPO evaluation.")
parser.add_argument("--checkpoint",   type=str, default=None)
parser.add_argument("--num_envs",     type=int, default=16)
parser.add_argument("--seed",         type=int, default=0)
parser.add_argument("--video",        action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import logging
import time

import gymnasium as gym
import numpy as np
import sbx

from isaaclab.utils.dict import print_dict
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


def _find_latest_checkpoint(log_root: str) -> str:
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log root not found: {log_root}")
    run_dirs = [os.path.join(log_root, d) for d in os.listdir(log_root)
                if os.path.isdir(os.path.join(log_root, d))]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories in: {log_root}")
    run_dir = max(run_dirs, key=os.path.getmtime)

    final = os.path.join(run_dir, "model.zip")
    if os.path.isfile(final):
        return final

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        zips = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
        if zips:
            return max(zips, key=os.path.getmtime)

    raise FileNotFoundError(f"No .zip checkpoint found in: {run_dir}")


def main():
    checkpoint_path = args_cli.checkpoint
    if checkpoint_path is None:
        checkpoint_path = _find_latest_checkpoint(LOG_ROOT)
        print(f"[INFO] Auto-selected checkpoint: {checkpoint_path}")

    env_cfg = UnitreeGo2WFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)                                                                                                                                       
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)                                                                                                                                       
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)                                                                                                                                       
    # env_cfg.commands.base_velocity.rel_standing_envs = 0.0  


    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        log_dir = os.path.join(LOG_ROOT, "eval")
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env, fast_variant=True)

    print(f"[INFO] Loading model: {checkpoint_path}")
    model = sbx.PPO.load(checkpoint_path, env=env)
    print("[INFO] Model loaded. Running until window is closed.\n")

    dt = env.unwrapped.step_dt
    obs = env.reset()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        actions, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
