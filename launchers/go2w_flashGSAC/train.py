"""Train Go2W rough terrain locomotion with FlashGSAC (Guided SAC + FlashSAC back-end).

FlashGSAC combines:
  - The FlashSAC distributional critic + zeta-noise exploration from Holiday-Robot/FlashSAC
  - The Guided SAC two-actor architecture (control actor on noisy obs, guide actor on
    clean privileged obs) from go2w_GSAC_sbx

The guide actor sees the critic observation group from Isaac Lab (no sensor noise).
The control actor sees the policy observation group (with standard noise).
Both are trained simultaneously; only the control actor is deployed at test time.
"""

# ─── Section 1: stdlib + sys.path setup ─────────────────────────────────────
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

_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

# FlashSAC library (adjacent sibling directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "go2w_flashsac", "FlashSAC"))

# flash_gsac package (this directory)
sys.path.insert(0, _SCRIPT_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

# ─── Section 2: CLI args + AppLauncher ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Go2W rough terrain with FlashGSAC.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0",
    help="IsaacLab gym task ID.",
)
parser.add_argument("--num_envs",     type=int,   default=256,          help="Parallel environments.")
parser.add_argument("--total_steps",  type=int,   default=50_000_000,   help="Total env steps.")
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--wandb_name",   type=str,   default="flashgsac_go2w")
parser.add_argument(
    "--guidance_weight", type=float, default=0.5,
    help="λ: weight of the L1 guidance loss on the control actor.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a checkpoint directory to resume from.",
)
parser.add_argument(
    "--guide_reward_alpha", type=float, default=1.0,
    help="α: weight of the height-gain bonus added to the guide actor's reward only.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _cleanup_on_interrupt(*_: Any) -> None:
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _cleanup_on_interrupt)

# ─── Section 3: post-launcher imports ───────────────────────────────────────
import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import wandb                     # noqa: E402
from gymnasium.vector.utils import batch_space  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

if _LAB_MACHINE:
    import QuadLoco  # type: ignore  # noqa: F401
else:
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401

from flash_gsac.agent import FlashGSACAgent, FlashGSACConfig  # noqa: E402

# ─── Section 4: GPU performance settings ────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ─── Section 5: Isaac Lab env wrapper ────────────────────────────────────────
ACTION_BOUNDS = 5.0  # Go2W-specific (matches go2w_SAC_sbx and go2w_flashsac)


class Go2WGuidedEnvWrapper:
    """
    Wraps a ManagerBasedRLEnv to expose the VectorEnv-like interface that
    FlashGSAC's training loop expects.

    Provides two observation streams per step:
      - policy obs   (noisy, shape=(n_envs, policy_obs_dim))  → control actor
      - critic obs   (clean, shape=(n_envs, critic_obs_dim))  → guide actor + critic

    The critic obs group must be registered in the env's observation manager
    (e.g. via CriticCfg with enable_corruption=False).
    """

    def __init__(self, env: Any, device: str) -> None:
        self._env      = env
        self._isaac_env = env.unwrapped
        self.device    = device
        self.num_envs: int = self._isaac_env.num_envs
        self.max_episode_steps: int = self._isaac_env.max_episode_length

        policy_obs_shape  = self._isaac_env.single_observation_space["policy"].shape
        critic_obs_shape  = self._isaac_env.observation_manager.group_obs_dim["critic"]
        action_shape      = self._isaac_env.single_action_space.shape

        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=policy_obs_shape, dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.guide_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=critic_obs_shape, dtype=np.float32
        )

        self.single_action_space = gym.spaces.Box(
            low=-ACTION_BOUNDS, high=ACTION_BOUNDS,
            shape=action_shape, dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    # ---------------------------------------------------------------------- #
    def _get_guide_obs(self) -> np.ndarray:
        """Read the clean critic obs from Isaac Lab's obs buffer (post-step/post-reset)."""
        return self._isaac_env.obs_buf["critic"].cpu().numpy().copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        random_start_init: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Returns
        -------
        policy_obs   : (n_envs, policy_obs_dim)
        guide_obs    : (n_envs, critic_obs_dim) — initial clean obs
        env_info     : dict
        """
        obs_dict, _ = self._isaac_env.reset(seed=seed)

        if random_start_init:
            self._isaac_env.episode_length_buf = torch.randint_like(
                self._isaac_env.episode_length_buf,
                high=int(self.max_episode_steps),
            )

        env_info: dict[str, Any] = {
            "actor_observation_size": self.single_observation_space.shape,
            "asymmetric_obs": False,
        }
        policy_obs = obs_dict["policy"].cpu().numpy()
        guide_obs  = self._get_guide_obs()
        return policy_obs, guide_obs, env_info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Returns
        -------
        policy_obs      : (n_envs, policy_obs_dim)
        guide_obs       : (n_envs, critic_obs_dim)  — post-step clean obs
        rewards         : (n_envs,)
        terminated      : (n_envs,)
        truncated       : (n_envs,)
        infos           : dict
        """
        torch_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * ACTION_BOUNDS

        obs_dict, rew, terminated, truncated, _ = self._isaac_env.step(torch_actions)

        policy_obs = obs_dict["policy"].cpu().numpy()
        # The critic obs buffer is updated every step (after auto-reset for done envs).
        # For done envs the guide_obs here is the post-reset guide obs, which is masked
        # by (1-done) in the Bellman target so it does not affect the TD update.
        guide_obs  = self._get_guide_obs()

        rew_np  = rew.cpu().numpy()
        term_np = terminated.cpu().numpy()
        trunc_np = truncated.cpu().numpy()

        infos: dict[str, Any] = {
            "final_obs": policy_obs.copy(),
            "time_outs": trunc_np,
        }
        return policy_obs, guide_obs, rew_np, term_np, trunc_np, infos

    def close(self) -> None:
        pass  # simulation_app.close() in __main__


# ─── Section 6: main ─────────────────────────────────────────────────────────

def main() -> None:
    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(
        os.path.join("logs", "flashgsac", args_cli.task,
                     f"{args_cli.wandb_name}_{run_timestamp}")
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir, "command.txt").write_text(" ".join(sys.argv))
    print(f"[FlashGSAC] Logging to: {log_dir}")

    # WandB
    wandb.init(
        project="flashgsac_go2w",
        name=args_cli.wandb_name,
        config={
            "task":             args_cli.task,
            "num_envs":         args_cli.num_envs,
            "total_steps":      args_cli.total_steps,
            "seed":             seed,
            "algorithm":        "FlashGSAC",
            "guidance_weight":  args_cli.guidance_weight,
            "action_bounds":    ACTION_BOUNDS,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    # ── Environment ──────────────────────────────────────────────────────────
    device_str: str = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.seed = seed
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = Go2WGuidedEnvWrapper(raw_env, device=device_str)

    print(
        f"[FlashGSAC] policy_obs shape: {env.single_observation_space.shape}  "
        f"guide_obs shape: {env.guide_observation_space.shape}  "
        f"act shape: {env.single_action_space.shape}  "
        f"num_envs: {env.num_envs}  "
        f"guidance_weight: {args_cli.guidance_weight}"
    )

    # ── FlashGSAC config ─────────────────────────────────────────────────────
    num_envs = env.num_envs
    updates_per_step = 2
    total_interaction_steps = args_cli.total_steps // num_envs
    total_update_steps      = total_interaction_steps * updates_per_step

    warmup_rate = 1e-6
    decay_rate  = 1.0
    warmup_step = max(1, int(warmup_rate * total_update_steps))
    decay_step  = max(1, int(decay_rate  * total_update_steps))

    agent_cfg = FlashGSACConfig(
        # ── Seeding ──
        seed=seed,
        # ── Reward normalisation ──
        normalize_reward=True,
        normalized_G_max=5.0,
        # ── Observation ──
        asymmetric_observation=False,
        device_type=device_str,
        # ── Buffer ──
        buffer_max_length=500_000,
        buffer_min_length=10_000,
        buffer_device_type="cuda",
        sample_batch_size=1024,
        # ── LR schedule ──
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=warmup_rate,
        learning_rate_warmup_step=warmup_step,
        learning_rate_decay_rate=decay_rate,
        learning_rate_decay_step=decay_step,
        # ── Control actor ──
        actor_num_blocks=2,
        actor_hidden_dim=128,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=16,
        actor_update_period=2,
        # ── Guide actor (same size as control actor) ──
        guide_actor_num_blocks=2,
        guide_actor_hidden_dim=128,
        # ── Critic ──
        critic_num_blocks=2,
        critic_hidden_dim=256,
        critic_num_bins=101,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        # ── Temperature ──
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=0.0,  # overridden by FlashGSACAgent.__init__
        # ── RL ──
        gamma=0.99,
        n_step=3,
        # ── GSAC guidance ──
        guidance_weight=args_cli.guidance_weight,
        # ── Performance ──
        use_compile=True,
        compile_mode="auto",
        use_amp=True,
        # ── Checkpoint loading (fresh start) ──
        load_optimizer=False,
        load_reward_normalizer=False,
    )

    # ── Initial reset ────────────────────────────────────────────────────────
    policy_obs, guide_obs, env_info = env.reset(random_start_init=True)
    prev_base_height = env._isaac_env.scene["robot"].data.root_pos_w[:, 2].cpu().numpy().copy()

    agent = FlashGSACAgent(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        guide_observation_space=env.guide_observation_space,
        env_info=env_info,
        cfg=agent_cfg,
    )
    print(f"[FlashGSAC] Agent ready. Target entropy: {agent._cfg.temp_target_entropy:.4f}")
    Path(log_dir, "config.json").write_text(
        json.dumps(dataclasses.asdict(agent._cfg), indent=2)
    )

    # Optionally restore from checkpoint
    if args_cli.checkpoint is not None:
        agent.load(args_cli.checkpoint)

    # ── Training loop constants ───────────────────────────────────────────────
    UPDATES_PER_STEP   = updates_per_step
    LOG_EVERY          = max(1, total_interaction_steps // 500)
    CHECKPOINT_EVERY   = max(1, total_interaction_steps // 50)

    transition: dict[str, Any] | None = None
    avg_meter: dict[str, list[float]] = {}
    start_time     = time.time()
    last_log_step  = 0
    last_ckpt_step = 0
    update_step    = 0

    ep_rew_buf = np.zeros(num_envs, dtype=np.float32)
    ep_len_buf = np.zeros(num_envs, dtype=np.int32)
    completed_ep_rews: list[float] = []
    completed_ep_lens: list[int]   = []

    # Track guide obs for the current time step (shifted each step).
    current_guide_obs = guide_obs  # guide obs at t=0 (from reset)

    try:
        for interaction_step in range(1, total_interaction_steps + 1):
            env_step = interaction_step * num_envs

            # ── Sample actions from control actor ────────────────────────────
            if agent.can_start_training() and transition is not None:
                actions: np.ndarray = agent.sample_actions(
                    interaction_step, transition, training=True
                )
            else:
                actions = env.action_space.sample()

            # ── Environment step ─────────────────────────────────────────────
            next_policy_obs, next_guide_obs, rewards, terminated, truncated, infos = env.step(actions)

            # ── Height-gain bonus for guide reward ───────────────────────────
            # Read base height after the step (post-reset for done envs, which is
            # fine — their guide_reward is masked by (1-done) in the Bellman target).
            next_base_height = env._isaac_env.scene["robot"].data.root_pos_w[:, 2].cpu().numpy()
            height_gain      = np.maximum(next_base_height - prev_base_height, 0.0)
            guide_reward     = rewards + args_cli.guide_reward_alpha * height_gain

            # ── Build buffer transition ──────────────────────────────────────
            done_mask = terminated | truncated
            next_buf_obs       = next_policy_obs.copy()
            next_buf_guide_obs = next_guide_obs.copy()
            if done_mask.any():
                next_buf_obs[done_mask] = infos["final_obs"][done_mask]

            transition = {
                "observation":            policy_obs,
                "action":                 actions,
                "reward":                 rewards,
                "terminated":             terminated,
                "truncated":              truncated,
                "next_observation":       next_buf_obs,
                "guide_observation":      current_guide_obs,
                "guide_next_observation": next_buf_guide_obs,
                "guide_reward":           guide_reward,
            }
            agent.process_transition(transition)

            # Keep next_observation pointing to the actual next obs (post-reset
            # for done envs) so sample_actions() gets the live obs.
            transition["next_observation"] = next_policy_obs

            # Shift guide obs and height forward for the next step.
            policy_obs        = next_policy_obs
            current_guide_obs = next_guide_obs
            prev_base_height  = next_base_height

            # ── Episode tracking ─────────────────────────────────────────────
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
                log_dict["env_step"]              = env_step
                log_dict["fps"]                   = env_step / elapsed
                log_dict["update_step"]           = update_step
                log_dict["guide/height_gain_mean"] = float(height_gain.mean())

                if completed_ep_rews:
                    log_dict["rollout/ep_rew_mean"] = float(np.mean(completed_ep_rews))
                    log_dict["rollout/ep_len_mean"] = float(np.mean(completed_ep_lens))
                    completed_ep_rews.clear()
                    completed_ep_lens.clear()

                # Isaac Lab reward breakdown + robot state
                try:
                    isaac_env = env._isaac_env
                    robot     = isaac_env.scene["robot"]
                    cmd       = isaac_env.command_manager.get_command("base_velocity")
                    lin_vel_b = robot.data.root_lin_vel_b[:, :2]
                    ang_vel_b = robot.data.root_ang_vel_b[:, 2]
                    grav_b    = robot.data.projected_gravity_b

                    log_dict["env/base_height"]             = float(robot.data.root_pos_w[:, 2].mean().item())
                    log_dict["env/lin_vel_tracking_error"]  = float((cmd[:, :2] - lin_vel_b).norm(dim=1).mean().item())
                    log_dict["env/ang_vel_tracking_error"]  = float((cmd[:, 2] - ang_vel_b).abs().mean().item())
                    log_dict["env/orientation_error"]       = float(grav_b[:, :2].norm(dim=1).mean().item())

                    if "log" in isaac_env.extras:
                        for key, value in isaac_env.extras["log"].items():
                            try:
                                log_dict[f"reward/{key}"] = float(
                                    value.mean().item() if isinstance(value, torch.Tensor) else value
                                )
                            except (ValueError, TypeError, AttributeError):
                                pass
                except (AttributeError, KeyError):
                    pass

                wandb.log(log_dict, step=env_step)
                avg_meter.clear()

                ep_rew_str = (
                    f"  ep_rew {log_dict['rollout/ep_rew_mean']:.1f}"
                    if "rollout/ep_rew_mean" in log_dict else ""
                )
                g_loss_str = (
                    f"  guidance {log_dict.get('actor/guidance_loss', 0):.4f}"
                    if "actor/guidance_loss" in log_dict else ""
                )
                print(
                    f"[FlashGSAC] step {env_step:>10,}  "
                    f"fps {log_dict.get('fps', 0):.0f}  "
                    f"updates {update_step:>8,}  "
                    f"elapsed {elapsed/60:.1f}min"
                    f"{ep_rew_str}{g_loss_str}"
                )

            # ── Checkpointing ─────────────────────────────────────────────────
            if interaction_step - last_ckpt_step >= CHECKPOINT_EVERY:
                last_ckpt_step = interaction_step
                ckpt_path = os.path.join(log_dir, f"step_{env_step}")
                agent.save(ckpt_path)

    except KeyboardInterrupt:
        print("\n[FlashGSAC] Training interrupted.")

    agent.save(os.path.join(log_dir, "final"))
    print(f"[FlashGSAC] Total training time: {(time.time() - start_time) / 60:.1f} min")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
