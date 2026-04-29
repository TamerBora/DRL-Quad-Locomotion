"""Train Go2W rough terrain locomotion with FlashSAC."""

# ─── Section 1: stdlib + sys.path setup ─────────────────────────────────────
# All non-stdlib imports happen AFTER AppLauncher launches below.
import argparse
import dataclasses
import json
import os
import sys
import signal
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any

# Inject robot_lab (same pattern as go2w_SAC_sbx/train.py)
_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

# Inject FlashSAC (cloned as a sibling subdirectory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "FlashSAC"))

from isaaclab.app import AppLauncher  # noqa: E402  (must be imported before Isaac Sim launches)

# ─── Section 2: CLI args + AppLauncher (MUST happen before any Isaac import) ─
parser = argparse.ArgumentParser(description="Train Go2W rough terrain with FlashSAC.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0",
    help="IsaacLab gym task ID.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--total_steps", type=int, default=50_000_000, help="Total environment steps.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--wandb_name", type=str, default="flashsacv1", help="WandB run name.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _cleanup_on_interrupt(*_: Any) -> None:
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _cleanup_on_interrupt)

# ─── Section 3: post-launcher imports ───────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from gymnasium.vector.utils import batch_space  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# Task registration — importing this module causes the gym.register() calls to run
if _LAB_MACHINE:
    import QuadLoco  # type: ignore  # noqa: F401
else:
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig  # noqa: E402

# ─── Section 4: RTX 4070 laptop torch settings ───────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ─── Section 5: Go2W IsaacLab environment wrapper ────────────────────────────
# Mirrors flash_rl/envs/isaaclab.py::IsaacLabVectorEnv exactly, but skips the
# internal AppLauncher call (Isaac Sim is already running at this point).
ACTION_BOUNDS = 5.0  # Go2W-specific: matches the SBX SAC pipeline (Box(-5,5) override in go2w_SAC_sbx/train.py)


class Go2WIsaacEnvWrapper:
    """
    Wraps a ManagerBasedRLEnv to expose the gymnasium VectorEnv-like interface
    that FlashSAC's training loop expects (numpy I/O, standard space attributes).
    """

    def __init__(self, env: Any, device: str) -> None:
        self._env = env
        self._isaac_env = env.unwrapped
        self.device = device
        self.num_envs: int = self._isaac_env.num_envs
        self.max_episode_steps: int = self._isaac_env.max_episode_length

        obs_size = self._isaac_env.single_observation_space["policy"].shape
        action_size = self._isaac_env.single_action_space.shape

        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=obs_size, dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-ACTION_BOUNDS,
            high=ACTION_BOUNDS,
            shape=action_size,
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, _ = self._isaac_env.reset(seed=seed)

        # Decorrelate episode horizons (avoids synchronised reset spikes)
        if random_start_init:
            self._isaac_env.episode_length_buf = torch.randint_like(
                self._isaac_env.episode_length_buf,
                high=int(self.max_episode_steps),
            )

        env_info: dict[str, Any] = {
            "actor_observation_size": self.single_observation_space.shape,
            "asymmetric_obs": False,
        }
        return obs_dict["policy"].cpu().numpy(), env_info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        torch_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        # Actor outputs in [-1,1] (tanh); scale to action_bounds before sending to Isaac
        torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * ACTION_BOUNDS

        obs_dict, rew, terminated, truncated, _ = self._isaac_env.step(torch_actions)

        obs_np = obs_dict["policy"].cpu().numpy()
        rew_np = rew.cpu().numpy()
        term_np = terminated.cpu().numpy()
        trunc_np = truncated.cpu().numpy()

        infos: dict[str, Any] = {
            "final_obs": obs_np.copy(),
            "time_outs": trunc_np,
        }
        return obs_np, rew_np, term_np, trunc_np, infos

    def close(self) -> None:
        pass  # simulation_app.close() is called in __main__


# ─── Section 6: main training function ───────────────────────────────────────

def main() -> None:
    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    WANDB_RUN_NAME = args_cli.wandb_name

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(
        os.path.join(
            "logs", "flashsac", args_cli.task, f"{WANDB_RUN_NAME}_{run_timestamp}"
        )
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir, "command.txt").write_text(" ".join(sys.argv))
    print(f"[FlashSAC] Logging to: {log_dir}")

    # WandB
    wandb.init(
        project="flashsac_go2w",
        name=WANDB_RUN_NAME,
        config={
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "total_steps": args_cli.total_steps,
            "seed": seed,
            "algorithm": "FlashSAC",
            "action_bounds": ACTION_BOUNDS,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    # Environment
    device_str: str = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.seed = seed
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = Go2WIsaacEnvWrapper(raw_env, device=device_str)

    print(
        f"[FlashSAC] obs shape: {env.single_observation_space.shape}  "
        f"act shape: {env.single_action_space.shape}  "
        f"num_envs: {env.num_envs}"
    )

    # FlashSAC config (RTX 4070 laptop — 8 GB VRAM)
    num_envs = env.num_envs
    updates_per_step = 2
    total_interaction_steps = args_cli.total_steps // num_envs
    total_update_steps = total_interaction_steps * updates_per_step

    warmup_rate = 1e-6
    decay_rate = 1.0
    warmup_step = max(1, int(warmup_rate * total_update_steps))
    decay_step = max(1, int(decay_rate * total_update_steps))

    agent_cfg = FlashSACConfig(
        seed=seed,
        normalize_reward=True,
        normalized_G_max=5.0,
        asymmetric_observation=False,
        device_type=device_str,
        # Buffer: 500K × ~obs_dim floats fits well within 8 GB
        buffer_max_length=500_000,
        buffer_min_length=10_000,
        buffer_device_type="cuda",
        sample_batch_size=1024,
        # Learning rate schedule
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=warmup_rate,
        learning_rate_warmup_step=warmup_step,
        learning_rate_decay_rate=decay_rate,
        learning_rate_decay_step=decay_step,
        # Actor (matches FlashSAC default)
        actor_num_blocks=2,
        actor_hidden_dim=128,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=16,
        actor_update_period=2,
        # Critic (matches FlashSAC default)
        critic_num_blocks=2,
        critic_hidden_dim=256,
        critic_num_bins=101,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        # Temperature
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=0.0,  # overridden by FlashSACAgent.__init__ from temp_target_sigma
        # RL
        gamma=0.99,
        n_step=3,
        # Performance
        use_compile=True,
        compile_mode="auto",  # resolves to reduce-overhead on torch < 2.9
        use_amp=True,         # essential for 8 GB VRAM
        # Checkpoint loading (fresh start)
        load_optimizer=False,
        load_reward_normalizer=False,
    )

    obs, env_info = env.reset(random_start_init=True)
    agent = FlashSACAgent(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env_info=env_info,
        cfg=agent_cfg,
    )
    print(f"[FlashSAC] Agent ready. Target entropy: {agent._cfg.temp_target_entropy:.4f}")
    Path(log_dir, "config.json").write_text(json.dumps(dataclasses.asdict(agent._cfg), indent=2))

    # Training loop constants
    UPDATES_PER_STEP = updates_per_step
    LOG_EVERY = max(1, total_interaction_steps // 500)       # ~500 log points
    CHECKPOINT_EVERY = max(1, total_interaction_steps // 50) # ~50 checkpoints

    transition: dict[str, Any] | None = None
    avg_meter: dict[str, list[float]] = {}
    start_time = time.time()
    last_log_step = 0
    last_ckpt_step = 0
    update_step = 0

    # Per-env episode accumulators
    ep_rew_buf = np.zeros(num_envs, dtype=np.float32)
    ep_len_buf = np.zeros(num_envs, dtype=np.int32)
    completed_ep_rews: list[float] = []
    completed_ep_lens: list[int] = []

    try:
        for interaction_step in range(1, total_interaction_steps + 1):
            env_step = interaction_step * num_envs

            # ── Collect actions ──────────────────────────────────────────────
            if agent.can_start_training() and transition is not None:
                actions: np.ndarray = agent.sample_actions(
                    interaction_step, transition, training=True
                )
            else:
                actions = env.action_space.sample()

            # ── Environment step ─────────────────────────────────────────────
            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            # ── Build buffer transition ──────────────────────────────────────
            # For done envs IsaacLab already returned the post-reset obs; use it
            # as "final_obs" (standard IsaacLab practice per FlashSAC's own wrapper).
            next_buf_obs = next_obs.copy()
            done_mask = terminated | truncated
            if done_mask.any():
                next_buf_obs[done_mask] = infos["final_obs"][done_mask]

            transition = {
                "observation": obs,
                "action": actions,
                "reward": rewards,
                "terminated": terminated,
                "truncated": truncated,
                "next_observation": next_buf_obs,
            }
            agent.process_transition(transition)

            # Overwrite next_observation with actual next obs for action sampling
            transition["next_observation"] = next_obs
            obs = next_obs

            # ── Episode reward / length tracking ─────────────────────────────
            ep_rew_buf += rewards
            ep_len_buf += 1
            if done_mask.any():
                completed_ep_rews.extend(ep_rew_buf[done_mask].tolist())
                completed_ep_lens.extend(ep_len_buf[done_mask].tolist())
                ep_rew_buf[done_mask] = 0.0
                ep_len_buf[done_mask] = 0

            # ── Gradient updates ─────────────────────────────────────────────
            if agent.can_start_training():
                for _ in range(UPDATES_PER_STEP):
                    step_metrics = agent.update()
                    update_step += 1
                    for k, v in step_metrics.items():
                        avg_meter.setdefault(k, []).append(float(v))

            # ── WandB logging ────────────────────────────────────────────────
            if interaction_step - last_log_step >= LOG_EVERY:
                last_log_step = interaction_step
                elapsed = time.time() - start_time

                log_dict: dict[str, Any] = {
                    k: float(np.mean(vals)) for k, vals in avg_meter.items() if vals
                }
                log_dict["env_step"] = env_step
                log_dict["fps"] = env_step / elapsed
                log_dict["update_step"] = update_step

                if completed_ep_rews:
                    log_dict["rollout/ep_rew_mean"] = float(np.mean(completed_ep_rews))
                    log_dict["rollout/ep_len_mean"] = float(np.mean(completed_ep_lens))
                    completed_ep_rews.clear()
                    completed_ep_lens.clear()

                # Isaac Lab reward breakdown and robot state
                try:
                    isaac_env = env._isaac_env
                    robot = isaac_env.scene["robot"]
                    cmd = isaac_env.command_manager.get_command("base_velocity")
                    lin_vel_b = robot.data.root_lin_vel_b[:, :2]
                    ang_vel_b = robot.data.root_ang_vel_b[:, 2]
                    grav_b = robot.data.projected_gravity_b

                    log_dict["env/base_height"] = float(
                        robot.data.root_pos_w[:, 2].mean().item()
                    )
                    log_dict["env/lin_vel_tracking_error"] = float(
                        (cmd[:, :2] - lin_vel_b).norm(dim=1).mean().item()
                    )
                    log_dict["env/ang_vel_tracking_error"] = float(
                        (cmd[:, 2] - ang_vel_b).abs().mean().item()
                    )
                    log_dict["env/orientation_error"] = float(
                        grav_b[:, :2].norm(dim=1).mean().item()
                    )

                    if "log" in isaac_env.extras:
                        for key, value in isaac_env.extras["log"].items():
                            try:
                                log_dict[f"reward/{key}"] = float(
                                    value.mean().item()
                                    if isinstance(value, torch.Tensor)
                                    else value
                                )
                            except (ValueError, TypeError, AttributeError):
                                pass
                except (AttributeError, KeyError):
                    pass

                wandb.log(log_dict, step=env_step)
                avg_meter.clear()

                ep_rew_str = (
                    f"  ep_rew {log_dict['rollout/ep_rew_mean']:.1f}"
                    if "rollout/ep_rew_mean" in log_dict
                    else ""
                )
                print(
                    f"[FlashSAC] step {env_step:>10,}  "
                    f"fps {log_dict.get('fps', 0):.0f}  "
                    f"updates {update_step:>8,}  "
                    f"elapsed {elapsed/60:.1f}min"
                    f"{ep_rew_str}"
                )

            # ── Checkpointing ────────────────────────────────────────────────
            if interaction_step - last_ckpt_step >= CHECKPOINT_EVERY:
                last_ckpt_step = interaction_step
                ckpt_path = os.path.join(log_dir, f"step_{env_step}")
                agent.save(ckpt_path)

    except KeyboardInterrupt:
        print("\n[FlashSAC] Training interrupted.")

    agent.save(os.path.join(log_dir, "final"))
    print(f"[FlashSAC] Total training time: {(time.time() - start_time) / 60:.1f} min")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
