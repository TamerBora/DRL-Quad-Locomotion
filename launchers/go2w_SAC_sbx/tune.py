#!/usr/bin/env python3
# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""
GO2W SBX — Optuna hyperparameter tuning with WandB logging.

Each invocation runs ONE trial (~8 minutes), stores result in SQLite DB.
Run this script N times to explore the search space:

    for i in $(seq 100); do
        ./isaaclab.sh -p /home/tamer/robotics/launchers/go2w_sbx/tune.py \\
            --num_envs 2048 --headless
    done

View results:
    optuna-dashboard sqlite:///go2w_sbx_optuna.db
    or on wandb.ai → project: go2w-sac-tuning
"""

import argparse
import os
import sys
import time
import random
import logging

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GO2W SBX — Optuna tuner.")
parser.add_argument("--num_envs",       type=int,   default=2048)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--trial_minutes",  type=float, default=8.0,  help="Max wall-clock minutes per trial.")
parser.add_argument("--study_name",     type=str,   default="go2w-sac-tuning")
parser.add_argument("--db_path",        type=str,   default=os.path.join(_SBX_DIR, "go2w_optuna.db"))
parser.add_argument("--wandb_project",  type=str,   default="go2w-sac-tuning")
parser.add_argument("--save_threshold", type=float, default=30.0, help="Save model artifact if reward exceeds this.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym
import numpy as np
import optax
import optuna
import jax.nn
import sbx
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.utils.io import dump_yaml

logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.WARNING)

if _LAB_MACHINE:
    import quadruped_lab   # noqa: F401
else:
    import robot_lab   # noqa: F401

if _LAB_MACHINE:
    from quadruped_lab.tasks.manager_based.locomotion.velocity.config.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg
else:
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg


# ── Pruner — kills bad trials at 3 checkpoints ───────────────────────────────
class StaircasePruner(optuna.pruners.BasePruner):
    """Prune trial if reward is below gate threshold at each time checkpoint."""
    # Gates: {checkpoint_index: min_reward_to_continue}
    # Adjust thresholds after seeing what rewards your robot achieves
    GATES = {0: 5.0, 1: 15.0, 2: 25.0}

    def prune(self, study, trial):
        step = trial.last_step
        if step is None or step not in self.GATES:
            return False
        return trial.intermediate_values[step] < self.GATES[step]


# ── Eval callback — reports reward to Optuna at timed checkpoints ─────────────
class TrialEvalCallback(BaseCallback):
    def __init__(self, trial, eval_env, trial_minutes: float):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.timeout = trial_minutes * 60
        self.start_time = time.time()
        # Evaluate at 30%, 60%, 95% of trial time
        frac = trial_minutes * 60
        self.checkpoints = [frac * 0.30, frac * 0.60, frac * 0.95]
        self.current_idx = 0
        self.is_pruned = False
        self.last_mean_reward = -999.0
        self.last_std_reward = 0.0

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time

        # Hard stop
        if elapsed >= self.timeout:
            return False

        if self.current_idx < len(self.checkpoints):
            if elapsed >= self.checkpoints[self.current_idx]:
                mean_reward, std_reward = evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=min(self.eval_env.num_envs, 256)
                )
                self.last_mean_reward = mean_reward
                self.last_std_reward = std_reward

                self.trial.report(mean_reward, self.current_idx)
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/checkpoint": self.current_idx,
                    "elapsed_seconds": elapsed,
                })
                print(f"[Gate {self.current_idx}] reward={mean_reward:.2f} ± {std_reward:.2f} at {elapsed:.0f}s")

                if self.trial.should_prune():
                    print(f"[PRUNED] Trial failed gate {self.current_idx} (reward={mean_reward:.2f})")
                    self.is_pruned = True
                    return False

                self.current_idx += 1

        return True


def build_hyperparams(trial: optuna.Trial) -> dict:
    """Sample hyperparameters from Optuna search space."""
    return {
        "gamma":           trial.suggest_float("gamma", 0.975, 0.995),
        "learning_rate":   trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "ent_coef":        f"auto_{trial.suggest_float('ent_coef_init', 0.001, 0.02, log=True)}",
        "batch_size":      2 ** trial.suggest_int("batch_size_pow", 7, 12),   # 128–4096
        "train_freq":      2 ** trial.suggest_int("train_freq_pow", 0, 3),    # 1–8
        "gradient_steps":  2 ** trial.suggest_int("gradient_steps_pow", 0, 10), # 1–1024
        "policy_delay":    2 ** trial.suggest_int("policy_delay_pow", 0, 5),  # 1–32
        "tau":             trial.suggest_float("tau", 0.001, 0.05, log=True),
        "buffer_size":     8_000_000,
        "learning_starts": 100,
        "policy_kwargs": {
            "layer_norm":      True,
            "net_arch":        [512, 256, 128],
            "activation_fn":   jax.nn.elu,
            "optimizer_class": optax.adamw,
        },
    }


def main():
    seed = args_cli.seed if args_cli.seed >= 0 else random.randint(0, 10_000)

    # ── Optuna study (shared SQLite DB across all trial invocations) ───────────
    storage_url = f"sqlite:///{args_cli.db_path}"
    study = optuna.create_study(
        study_name=args_cli.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=StaircasePruner(),
    )
    trial = study.ask()
    hyperparams = build_hyperparams(trial)

    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number} — {args_cli.study_name}")
    print(f"{'='*60}")
    for k, v in trial.params.items():
        print(f"  {k:25s}: {v}")
    print(f"{'='*60}\n")

    # ── WandB run for this trial ───────────────────────────────────────────────
    wandb_run = wandb.init(
        project=args_cli.wandb_project,
        name=f"trial_{trial.number}",
        config={**trial.params, "num_envs": args_cli.num_envs, "seed": seed},
        reinit=True,
    )

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = UnitreeGo2WFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = seed

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)
    env.action_space = gym.spaces.Box(
        low=-2.0, high=2.0, shape=env.action_space.shape, dtype=np.float32
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    log_dir = os.path.join(_SBX_DIR, "logs", "optuna", f"trial_{trial.number}")
    model = sbx.SAC(
        "MlpPolicy", env,
        seed=seed,
        tensorboard_log=log_dir,
        verbose=0,
        **hyperparams,
    )

    eval_cb = TrialEvalCallback(trial, env, args_cli.trial_minutes)

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=int(1e8),
            callback=[eval_cb],
            progress_bar=True,
            log_interval=500,
        )

        final_reward = eval_cb.last_mean_reward

        if not eval_cb.is_pruned:
            study.tell(trial, final_reward)
            print(f"[Trial {trial.number}] COMPLETE — reward={final_reward:.2f}")

            # Save model artifact to WandB if reward is good
            if final_reward >= args_cli.save_threshold:
                save_dir = os.path.join(_SBX_DIR, "logs", "optuna", "successes")
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"trial_{trial.number}_reward{final_reward:.1f}")
                model.save(model_path)
                print(f"[Trial {trial.number}] Model saved: {model_path}.zip")

                artifact = wandb.Artifact(
                    name=f"model-trial{trial.number}",
                    type="model",
                    description=f"Go2W SAC trial {trial.number}, reward={final_reward:.2f}",
                    metadata=trial.params,
                )
                artifact.add_file(model_path + ".zip")
                wandb_run.log_artifact(artifact)
                print(f"[Trial {trial.number}] Model uploaded to WandB artifacts.")

        else:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)

    except Exception as e:
        print(f"[ERROR] Trial {trial.number}: {e}")
        study.tell(trial, state=optuna.trial.TrialState.FAIL)

    finally:
        wandb.log({"trial/final_reward": eval_cb.last_mean_reward, "trial/number": trial.number})
        wandb.finish()
        env.close()

    # ── Print study summary ───────────────────────────────────────────────────
    try:
        best = study.best_trial
        print(f"\n[Study] Best so far: trial {best.number}, reward={best.value:.2f}")
        print(f"[Study] Best params: {best.params}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
    simulation_app.close()
