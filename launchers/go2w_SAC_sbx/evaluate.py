"""
Evaluate a Go2W checkpoint across 4 terrain types simultaneously.

Robots are split into 4 equal groups, each assigned to a specific terrain type:
  0 — Flat terrain
  1 — Uneven / random rough
  2 — Stairs (ascending)
  3 — Stairs (descending)

Terrain type assignment is deterministic (see TerrainImporter._compute_env_origins_curriculum):
  terrain_type[i] = floor(i / (num_envs / num_cols))
So group sizes are exactly num_envs // 4.

Usage:
  python go2w_SAC_sbx/evaluate.py \\
    --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \\
    --algorithm ppo \\
    --ml_framework jax \\
    --checkpoint logs/sb3/.../model_*.zip \\
    [--num_envs 32]          # must be a multiple of 4
    [--eval_steps 3000]      # steps per terrain group before reporting
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Go2W checkpoint across terrain types.")
parser.add_argument("--task",        type=str, required=True, help="Gym task ID.")
parser.add_argument("--algorithm",   type=str, required=True, choices=["ppo", "sac"])
parser.add_argument("--ml_framework",type=str, default="jax", choices=["jax", "torch"])
parser.add_argument("--checkpoint",  type=str, required=True, help="Path to .zip checkpoint.")
parser.add_argument("--num_envs",    type=int, default=32,
                    help="Total environments — must be divisible by 4 (8 per terrain type).")
parser.add_argument("--eval_steps",  type=int, default=3000,
                    help="Simulation steps to run before computing final metrics.")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--cmd",         type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
                    help="Velocity command applied to all robots (default: 1.0 0.0 0.0).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

assert args_cli.num_envs % 4 == 0, f"--num_envs must be divisible by 4, got {args_cli.num_envs}"

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import gymnasium as gym
import logging
from stable_baselines3.common.vec_env import VecNormalize

if args_cli.ml_framework == "torch":
    if args_cli.algorithm == "ppo":
        from stable_baselines3 import PPO as RLAlgorithm
    else:
        from stable_baselines3 import SAC as RLAlgorithm
else:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    import jax.nn
    if args_cli.algorithm == "ppo":
        from sbx import PPO as RLAlgorithm
    else:
        from sbx import SAC as RLAlgorithm

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

if _LAB_MACHINE:
    import quadruped_lab.tasks  # noqa: F401
else:
    import isaaclab_tasks.manager_based.locomotion.velocity.config.a1  # noqa: F401
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRoughEnvCfg,
)

# ── Evaluation terrain: 4 terrain types, num_rows=1 (no difficulty curriculum) ──
#
# terrain_type[i] = floor(i / (num_envs / 4))
# → robots split into 4 equal groups, one per terrain column.
# num_rows=1 forces all robots to start at the only available difficulty level.

EVAL_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,      # single difficulty row — no curriculum pressure during eval
    num_cols=4,      # 4 terrain type columns
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Column 0 — flat
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.25,
        ),
        # Column 1 — uneven rough
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.02, 0.07),
            noise_step=0.02,
            border_width=0.25,
        ),
        # Column 2 — ascending stairs
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.10),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Column 3 — descending stairs
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.10),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

TERRAIN_NAMES = ["Flat", "Uneven", "Stairs Up", "Stairs Down"]


@configclass
class UnitreeGo2WEvalEnvCfg(UnitreeGo2WRoughEnvCfg):
    """Evaluation environment: 4 terrain types in parallel, no curriculum."""

    def __post_init__(self):
        super().__post_init__()

        # Replace training terrain with fixed 4-column eval terrain
        self.scene.terrain.terrain_generator = EVAL_TERRAIN_CFG

        # Disable terrain curriculum — robots stay on their assigned terrain type
        self.curriculum.terrain_levels = None

        # Longer episodes for evaluation (don't cut short)
        self.episode_length_s = 20.0

        if self.__class__.__name__ == "UnitreeGo2WEvalEnvCfg":
            self.disable_zero_weight_rewards()


# ── Main ──────────────────────────────────────────────────────────────────────

_agent_entry_point = f"sb3_{args_cli.algorithm.lower()}_cfg_entry_point"

@hydra_task_config(args_cli.task, _agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    num_envs    = args_cli.num_envs
    group_size  = num_envs // 4
    checkpoint  = args_cli.checkpoint

    # Override env config with eval settings
    eval_cfg = UnitreeGo2WEvalEnvCfg()
    eval_cfg.scene.num_envs = num_envs
    eval_cfg.seed = args_cli.seed
    eval_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    print(f"\n{'='*60}")
    print(f"  Go2W Terrain Evaluation")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Num envs   : {num_envs}  ({group_size} per terrain)")
    print(f"  Eval steps : {args_cli.eval_steps}")
    print(f"  Command    : lin_x={args_cli.cmd[0]}, lin_y={args_cli.cmd[1]}, ang_z={args_cli.cmd[2]}")
    print(f"{'='*60}\n")

    # ── Build env ────────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=eval_cfg)

    # Verify terrain type assignment
    terrain_importer = env.unwrapped.scene.terrain
    terrain_types = terrain_importer.terrain_types.cpu().numpy()  # (num_envs,)
    for t_idx, name in enumerate(TERRAIN_NAMES):
        group = np.where(terrain_types == t_idx)[0]
        print(f"  Terrain {t_idx} ({name:12s}): env indices {group[0]}–{group[-1]} ({len(group)} envs)")
    print()

    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # Load VecNormalize stats if available
    vec_norm_path = Path(checkpoint.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl"))
    if vec_norm_path.exists():
        print(f"Loading VecNormalize stats: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False
        env.norm_reward = False

    agent = RLAlgorithm.load(checkpoint, env, print_system_info=False)
    print(f"Checkpoint loaded.\n")

    # Fixed velocity command
    cmd_override = torch.tensor(args_cli.cmd, device=eval_cfg.sim.device).unsqueeze(0).expand(num_envs, -1)

    # ── Per-group metric accumulators ─────────────────────────────────────────
    # episode_returns[t][i] = list of episode returns for terrain t, env i
    ep_returns_per_group  = [[] for _ in range(4)]
    ep_lengths_per_group  = [[] for _ in range(4)]
    falls_per_group       = [[] for _ in range(4)]  # True = fall, False = timeout

    # Running accumulators per env
    ep_return = np.zeros(num_envs)
    ep_len    = np.zeros(num_envs, dtype=int)

    obs = env.reset()

    for step in range(args_cli.eval_steps):
        # Inject fixed command
        env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd_override

        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        ep_return += rewards
        ep_len    += 1

        for env_i in range(num_envs):
            if dones[env_i]:
                t_idx = int(terrain_types[env_i])
                ep_returns_per_group[t_idx].append(float(ep_return[env_i]))
                ep_lengths_per_group[t_idx].append(int(ep_len[env_i]))

                # Fall = terminated (not truncated by time limit)
                is_truncated = infos[env_i].get("TimeLimit.truncated", False)
                falls_per_group[t_idx].append(not is_truncated)

                ep_return[env_i] = 0.0
                ep_len[env_i]    = 0

        # Print progress every 500 steps
        if (step + 1) % 500 == 0:
            ep_counts = [len(ep_returns_per_group[t]) for t in range(4)]
            print(f"  Step {step+1:>5}/{args_cli.eval_steps}  |  Episodes: "
                  + "  ".join(f"{TERRAIN_NAMES[t]}: {ep_counts[t]}" for t in range(4)))

    env.close()

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Terrain':<16} {'Episodes':>8} {'Mean Return':>12} {'Mean Steps':>11} {'Fall Rate':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*12} {'-'*11} {'-'*10}")

    for t_idx, name in enumerate(TERRAIN_NAMES):
        returns  = ep_returns_per_group[t_idx]
        lengths  = ep_lengths_per_group[t_idx]
        falls    = falls_per_group[t_idx]

        if len(returns) == 0:
            print(f"  {name:<16} {'0':>8}  (no episodes completed)")
            continue

        mean_ret  = np.mean(returns)
        mean_len  = np.mean(lengths)
        fall_rate = np.mean(falls) * 100.0

        print(f"  {name:<16} {len(returns):>8} {mean_ret:>12.2f} {mean_len:>11.1f} {fall_rate:>9.1f}%")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
