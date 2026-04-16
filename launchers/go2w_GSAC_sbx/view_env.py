"""Spawn the Go2W rough terrain environment with zero actions for visual inspection."""

import argparse
import os
import sys

_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
if not _LAB_MACHINE:
    _ROBOTICS_DIR = os.path.expanduser("~/robotics")
    sys.path.insert(0, os.path.join(_ROBOTICS_DIR, "robot_lab", "source", "robot_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View Go2W rough terrain environment.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np

if _LAB_MACHINE:
    import QuadLoco  # noqa: F401
else:
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "sb3_sac_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable terrain curriculum so all difficulty levels spawn at once for inspection
    env_cfg.curriculum.terrain_levels = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    action_dim = env.action_space.shape[0]
    zero_actions = np.zeros((args_cli.num_envs, action_dim), dtype=np.float32)

    print(f"[INFO] Task:           {args_cli.task}")
    print(f"[INFO] Num envs:       {args_cli.num_envs}")
    print(f"[INFO] Obs space:      {env.observation_space}")
    print(f"[INFO] Action space:   {env.action_space}")
    print("[INFO] Stepping with zero actions — use Isaac Sim viewport to inspect terrain.")
    print("[INFO] Press Ctrl+C to exit.")

    env.reset()
    while simulation_app.is_running():
        env.step(zero_actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
