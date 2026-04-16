"""Script to play a GSAC checkpoint on Go2W."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

# Auto-detect lab machine (has quadruped_lab); everyone else uses robot_lab.
# Set ROBOT_LAB_DIR to override the default robot_lab location.
_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a GSAC checkpoint on Go2W.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit path to a .zip checkpoint. "
                         "If omitted, auto-selects best_model.zip from the latest log dir.")
parser.add_argument("--use_last", action="store_true", default=False,
                    help="Load the last model.zip instead of best_model.zip.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("LIN_X", "LIN_Y", "ANG_Z"),
                    help="Override velocity command, e.g. --cmd 1.0 0.0 0.0")
parser.add_argument("--random_cmd", action="store_true", default=False,
                    help="Randomize 2D velocity commands (lin_x, lin_y, ang_z) periodically.")
parser.add_argument("--cmd_interval", type=float, default=5.0,
                    help="Seconds between command resamples when --random_cmd is set.")
parser.add_argument("--keep_all_info", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import random
import time

import gymnasium as gym
import numpy as np
import torch

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)

from sbx.gsac import GSAC
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

if _LAB_MACHINE:
    import QuadLoco  # noqa: F401
else:
    import isaaclab_tasks.manager_based.locomotion.velocity.config.a1  # noqa: F401
    import robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_a1  # noqa: F401
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _find_checkpoint(log_root: str, prefer_best: bool) -> str:
    """Auto-find the best_model.zip or model.zip in the most recent run dir."""
    log_root = Path(log_root)
    if not log_root.exists():
        raise FileNotFoundError(f"Log root not found: {log_root}")

    # Most recent run = last alphabetically (runs are timestamped)
    run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {log_root}")
    run_dir = run_dirs[-1]

    target = "model.zip" if not prefer_best else "best_model.zip"
    candidate = run_dir / target
    if not candidate.exists():
        # fallback: try the other one
        fallback = run_dir / ("best_model.zip" if not prefer_best else "model.zip")
        if fallback.exists():
            print(f"[INFO] {target} not found, falling back to {fallback.name}")
            candidate = fallback
        else:
            raise FileNotFoundError(
                f"Neither best_model.zip nor model.zip found in {run_dir}"
            )
    return str(candidate)


@hydra_task_config(args_cli.task, "sb3_sac_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play Go2W with a GSAC checkpoint."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve checkpoint path
    if args_cli.checkpoint is not None:
        checkpoint_path = args_cli.checkpoint
    else:
        log_root = os.path.abspath(
            os.path.join("logs", "gsac", args_cli.task)
        )
        checkpoint_path = _find_checkpoint(log_root, prefer_best=not args_cli.use_last)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        })

    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # Auto-detect spaces from checkpoint
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(checkpoint_path)
        if "action_space" in data:
            env.action_space = data["action_space"]
            print(f"[INFO] Action space (from checkpoint): {env.action_space}")
        if "observation_space" in data:
            env.observation_space = data["observation_space"]
            print(f"[INFO] Observation space (from checkpoint): {env.observation_space}")
    except Exception as e:
        print(f"[WARNING] Space auto-detection failed: {e}")

    # Load VecNormalize stats if they exist
    vec_norm_path = Path(checkpoint_path).parent / "model_vecnormalize.pkl"
    if vec_norm_path.exists():
        print(f"[INFO] Loading VecNormalize stats: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False
        env.norm_reward = False

    # Load GSAC
    agent = GSAC.load(checkpoint_path, env, print_system_info=True)
    print("[INFO] GSAC model loaded. Running inference with control actor.")

    dt = env.unwrapped.step_dt
    cmd_override = None
    if args_cli.cmd is not None:
        cmd_override = (
            torch.tensor(args_cli.cmd, device=env.unwrapped.device)
            .unsqueeze(0)
            .expand(env.unwrapped.num_envs, -1)
        )
        print(f"[INFO] Command override: lin_x={args_cli.cmd[0]}, lin_y={args_cli.cmd[1]}, ang_z={args_cli.cmd[2]}")

    obs = env.reset()
    timestep = 0
    last_resample_time = time.time()
    random_cmd = None

    def _sample_cmd(num_envs, device, ang_z_val=0.0):
        lin_x = np.random.uniform(-1.0, 1.0, size=(num_envs,))
        lin_y = np.random.uniform(-1.0, 1.0, size=(num_envs,))
        ang_z = np.full((num_envs,), ang_z_val)
        cmd = np.stack([lin_x, lin_y, ang_z], axis=1)
        return torch.tensor(cmd, dtype=torch.float32, device=device)

    if args_cli.random_cmd:
        random_cmd = _sample_cmd(env.unwrapped.num_envs, env.unwrapped.device, ang_z_val=0.0)
        print(f"[INFO] Random 2D commands enabled (ang_z=0 fixed), resampling every {args_cli.cmd_interval}s")

    while simulation_app.is_running():
        start = time.time()

        if cmd_override is not None:
            env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd_override
        elif args_cli.random_cmd:
            now = time.time()
            if now - last_resample_time >= args_cli.cmd_interval:
                random_cmd = _sample_cmd(env.unwrapped.num_envs, env.unwrapped.device, ang_z_val=0.0)
                last_resample_time = now
            env.unwrapped.command_manager.get_command("base_velocity")[:] = random_cmd

        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
