"""Script to train Go2W with Guided Soft Actor-Critic (GSAC)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import signal
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Auto-detect lab machine (has quadruped_lab); everyone else uses robot_lab.
# Set ROBOT_LAB_DIR to override the default robot_lab location.
_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

import optax

from isaaclab.app import AppLauncher
import torch

parser = argparse.ArgumentParser(description="Train Go2W with Guided SAC (GSAC).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--video_interval", type=int, default=50_000_000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0")
parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point",
                    help="Agent config entry point (reuses SAC yaml).")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--log_interval", type=int, default=100_000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--keep_all_info", action="store_true", default=False)
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
parser.add_argument("--wandb_name", type=str, default="gsac_go2w")
# GSAC-specific
parser.add_argument("--guidance_weight", type=float, default=0.5,
                    help="λ: weight of L1 guidance loss on control actor.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    import gc
    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import logging
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)
import jax.nn

from sbx.gsac import GSAC

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

if _LAB_MACHINE:
    import QuadLoco  # noqa: F401
else:
    import isaaclab_tasks.manager_based.locomotion.velocity.config.a1  # noqa: F401
    import robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_a1  # noqa: F401
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

import wandb
from wandb.integration.sb3 import WandbCallback


class GSACLoggerCallback(BaseCallback):
    """Logs Isaac Lab native metrics + GSAC-specific losses to WandB."""

    def __init__(self, log_dir: str = "", scalar_freq: int = 1, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.scalar_freq = scalar_freq

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        log_dict = {}

        if actions is not None and (self.n_calls == 1 or self.n_calls % self.scalar_freq == 0):
            actions_np = np.asarray(actions)
            log_dict.update({
                "actions/abs_mean_action": float(np.mean(np.abs(actions_np))),
                "actions/max_action": float(np.max(actions_np)),
                "actions/min_action": float(np.min(actions_np)),
            })

        # Isaac Lab native metrics
        try:
            vec_env = self.training_env
            isaac_env = vec_env.unwrapped
            if hasattr(isaac_env, "unwrapped"):
                isaac_env = isaac_env.unwrapped
            if "log" in isaac_env.extras:
                for key, value in isaac_env.extras["log"].items():
                    try:
                        val = float(value.mean().item()) if isinstance(value, torch.Tensor) else float(value)
                        log_dict[f"isaac_native/{key}"] = val
                    except (ValueError, TypeError):
                        continue
            robot = isaac_env.scene["robot"]
            grav_b = robot.data.projected_gravity_b
            cmd = isaac_env.command_manager.get_command("base_velocity")
            lin_vel_b = robot.data.root_lin_vel_b[:, :2]
            log_dict["env/orientation_error"]       = float(grav_b[:, :2].norm(dim=1).mean().item())
            log_dict["env/lin_vel_tracking_error"]  = float((cmd[:, :2] - lin_vel_b).norm(dim=1).mean().item())
            log_dict["env/ang_vel_tracking_error"]  = float((cmd[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs().mean().item())
            log_dict["env/base_height"]             = float(robot.data.root_pos_w[:, 2].mean().item())
            log_dict["env/roll_error_rad"]          = float(grav_b[:, 1].abs().mean().item())
            log_dict["env/pitch_error_rad"]         = float(grav_b[:, 0].abs().mean().item())
        except (AttributeError, KeyError):
            pass

        if log_dict:
            for key, val in log_dict.items():
                if isinstance(val, float):
                    self.logger.record(key, val)
            log_dict["timestep"] = self.num_timesteps
            wandb.log(log_dict, commit=False)

        return True


class BestModelCallback(BaseCallback):
    """
    Saves best_model.zip whenever mean episode reward improves.

    Uses completed episode returns from the training env — no separate
    eval env needed. "Best" = highest mean episode reward over the last
    `window` completed episodes across all parallel envs.
    """

    def __init__(self, save_path: str, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.window = window
        self._ep_rewards: list[float] = []
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # SB3 stores completed episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self._ep_rewards.append(float(ep_info["r"]))

        if len(self._ep_rewards) >= self.window:
            mean_reward = float(np.mean(self._ep_rewards[-self.window:]))
            if mean_reward > self._best_mean_reward:
                self._best_mean_reward = mean_reward
                path = os.path.join(self.save_path, "best_model")
                self.model.save(path)
                if self.verbose:
                    print(f"[BestModel] New best mean reward: {mean_reward:.2f} → saved to {path}.zip")
                wandb.log({"best/mean_reward": mean_reward,
                           "best/timestep": self.num_timesteps}, commit=False)
        return True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train Go2W with GSAC."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.wandb_name:
        run_info = f"{args_cli.wandb_name}_{run_info}"
    log_root_path = os.path.abspath(os.path.join("logs", "gsac", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))

    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    run = wandb.init(
        project="comparisons",
        name=args_cli.wandb_name,
        config={
            "guidance_weight": args_cli.guidance_weight,
            "task": args_cli.task,
            "algorithm": "GSAC",
            **vars(env_cfg),
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # Go2W action space
    env.action_space = gym.spaces.Box(
        low=-5,
        high=5,
        shape=(16,),
        dtype=np.float32,
    )
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Guidance weight λ: {args_cli.guidance_weight}")

    # Activation fn
    agent_cfg["policy_kwargs"]["activation_fn"] = jax.nn.elu
    if "optimizer_class" in agent_cfg["policy_kwargs"]:
        if agent_cfg["policy_kwargs"]["optimizer_class"] and \
           str(agent_cfg["policy_kwargs"]["optimizer_class"]).lower() == "optax.adamw":
            agent_cfg["policy_kwargs"]["optimizer_class"] = optax.adamw

    # Normalisation
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)
    if norm_args and norm_args.get("normalize_input"):
        print(f"[INFO] Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    checkpoint_interval = agent_cfg.pop("checkpoint_interval")

    # ── Privileged observations for the guide actor ─────────────────────────
    # The critic obs group (CriticCfg) has enable_corruption=False — it reads
    # the same state terms as the policy obs but with no sensor noise.
    # We expose it to the guide actor so the guide sees ground-truth signals
    # while the control actor only ever sees the noisy policy observations.
    #
    # Isaac Lab stores ALL observation groups (policy + critic) in env.obs_buf
    # after every step (ManagerBasedRLEnv.obs_buf = observation_manager.compute()).
    # extras["observations"] is NOT populated — obs_buf is the correct source.
    isaac_env = env.unwrapped
    if hasattr(isaac_env, "unwrapped"):
        isaac_env = isaac_env.unwrapped
    critic_obs_dim = isaac_env.observation_manager.group_obs_dim["critic"]
    guide_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=critic_obs_dim, dtype=np.float32)

    def get_guide_obs_fn(_obs: np.ndarray, _vec_env) -> np.ndarray:
        # obs_buf["critic"] is updated every step by observation_manager.compute().
        # Shape: (n_envs, critic_dim). The .copy() in collect_rollouts handles timing.
        return isaac_env.obs_buf["critic"].cpu().numpy()

    print(f"[INFO] Guide observation space (critic, no noise): {guide_obs_space}")

    # ── Create GSAC agent ────────────────────────────────────────────────────
    agent = GSAC(
        policy_arch,
        env,
        guide_observation_space=guide_obs_space,
        get_guide_obs_fn=get_guide_obs_fn,
        guidance_weight=args_cli.guidance_weight,
        verbose=0,
        tensorboard_log=log_dir,
        **agent_cfg,
    )

    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=log_dir,
        name_prefix="model",
        verbose=2,
    )
    callbacks = [
        checkpoint_callback,
        BestModelCallback(save_path=log_dir, window=100, verbose=1),
        GSACLoggerCallback(log_dir=log_dir),
        LogEveryNTimesteps(n_steps=args_cli.log_interval),
        WandbCallback(gradient_save_freq=10, verbose=2),
    ]

    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10,
        )

    agent.save(os.path.join(log_dir, "model"))
    print(f"[INFO] Saved model to: {os.path.join(log_dir, 'model.zip')}")

    if isinstance(env, VecNormalize):
        print("[INFO] Saving normalization stats")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    print(f"[INFO] Training time: {round(time.time() - start_time, 2)}s")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
