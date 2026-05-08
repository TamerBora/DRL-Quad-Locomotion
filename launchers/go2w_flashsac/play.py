"""Play a trained FlashSAC checkpoint on the Go2W task."""

# ─── Section 1: stdlib + sys.path setup ─────────────────────────────────────
import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "FlashSAC"))

from isaaclab.app import AppLauncher  # noqa: E402

# ─── Section 2: CLI args + AppLauncher ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Play a trained FlashSAC Go2W checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0",
    help="IsaacLab gym task ID.",
)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a specific checkpoint directory (contains actor.pt). "
                         "If omitted, auto-discovers the latest one under logs/flashsac/.")
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of parallel envs for visualisation.")
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Stop after this many completed episodes.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--real_time", action="store_true", default=False,
                    help="Throttle stepping to real-time based on sim dt.")
parser.add_argument("--cmd", type=float, nargs=3, default=None,
                    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
                    help="Override velocity command every step, e.g. --cmd 1.0 0.0 0.0")
parser.add_argument("--camera_switch_interval", type=float, default=0.0,
                    help="If >0 and num_envs>1, cycle the viewport camera to the next "
                         "robot every N real-time seconds. 0 = stay on env 0.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Section 3: post-launcher imports ───────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium.vector.utils import batch_space  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

if _LAB_MACHINE:
    import QuadLoco  # type: ignore  # noqa: F401
else:
    import robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w  # noqa: F401

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig  # noqa: E402

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ACTION_BOUNDS = 5.0  # Go2W-specific: matches the SBX SAC pipeline (Box(-5,5) override in go2w_SAC_sbx/train.py)


# ─── Section 4: checkpoint discovery ────────────────────────────────────────

def _find_latest_checkpoint(task: str) -> Path:
    """Return the most recent step_N or final dir under logs/flashsac/<task>/."""
    logs_root = Path(_SCRIPT_DIR) / "logs" / "flashsac" / task
    if not logs_root.exists():
        raise FileNotFoundError(
            f"No checkpoint found under {logs_root}.\n"
            "Run train.py first or pass --checkpoint explicitly."
        )

    # Collect all step_* and final directories across all run subdirs
    candidates: list[Path] = []
    for run_dir in sorted(logs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for ckpt_dir in run_dir.iterdir():
            if ckpt_dir.is_dir() and (ckpt_dir.name.startswith("step_") or ckpt_dir.name == "final"):
                if (ckpt_dir / "actor.pt").exists():
                    candidates.append(ckpt_dir)

    if not candidates:
        raise FileNotFoundError(
            f"Found run dirs under {logs_root} but no checkpoint subdirectories with actor.pt."
        )

    # Sort: prefer 'final', otherwise highest step number, break ties by mtime
    def _sort_key(p: Path) -> tuple[int, int, float]:
        is_final = int(p.name == "final")
        step_num = int(p.name.split("_")[1]) if p.name.startswith("step_") else 0
        return (is_final, step_num, p.stat().st_mtime)

    latest = sorted(candidates, key=_sort_key)[-1]
    return latest


def _default_config() -> FlashSACConfig:
    """Fallback config matching the RTX 4070 defaults in train.py."""
    return FlashSACConfig(
        seed=42,
        normalize_reward=True,
        normalized_G_max=5.0,
        asymmetric_observation=False,
        device_type="cuda",
        buffer_max_length=500_000,
        buffer_min_length=10_000,
        buffer_device_type="cuda",
        sample_batch_size=1024,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=1,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=1,
        actor_num_blocks=2,
        actor_hidden_dim=128,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=16,
        actor_update_period=2,
        critic_num_blocks=2,
        critic_hidden_dim=256,
        critic_num_bins=101,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=0.0,
        gamma=0.99,
        n_step=3,
        use_compile=True,
        compile_mode="auto",
        use_amp=True,
        load_optimizer=False,
        load_reward_normalizer=False,
    )


def _load_config(ckpt_dir: Path) -> FlashSACConfig:
    """Load FlashSACConfig from config.json, falling back to train.py defaults."""
    run_dir = ckpt_dir.parent if ckpt_dir.name == "final" or ckpt_dir.name.startswith("step_") else ckpt_dir
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"[FlashSAC] config.json not found at {config_path} — using default architecture config.")
        return _default_config()
    raw = json.loads(config_path.read_text())
    return FlashSACConfig(**raw)


def _infer_actor_input_dim(actor_path: Path) -> int | None:
    """Read the actor's expected obs dim from its BatchNorm running_mean shape.

    Returns None if the key isn't found (e.g. a new architecture without
    UnitBatchNorm at the embedder).
    """
    try:
        ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
    except (FileNotFoundError, RuntimeError):
        return None
    state = ckpt.get("network_state_dict", ckpt)
    for key, tensor in state.items():
        # Match either 'embedder.norm.running_mean' or compiled-module
        # variants like '_orig_mod.embedder.norm.running_mean'.
        if key.endswith("embedder.norm.running_mean"):
            return int(tensor.shape[0])
    return None


# ─── Section 5: env wrapper (same interface as train.py) ────────────────────

class Go2WIsaacEnvWrapper:
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
            low=-ACTION_BOUNDS, high=ACTION_BOUNDS, shape=action_size, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        # Go through self._env so any wrappers (e.g. RecordVideo when
        # --video is set) see the reset and start a new recording.
        obs_dict, _ = self._env.reset()
        env_info: dict[str, Any] = {
            "actor_observation_size": self.single_observation_space.shape,
            "asymmetric_obs": False,
        }
        return obs_dict["policy"].cpu().numpy(), env_info

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        torch_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * ACTION_BOUNDS
        # Go through self._env (wrapped chain) so RecordVideo sees each step
        # and captures a frame; self._isaac_env is the unwrapped base and
        # would skip the wrapper.
        obs_dict, rew, terminated, truncated, _ = self._env.step(torch_actions)
        return (
            obs_dict["policy"].cpu().numpy(),
            rew.cpu().numpy(),
            terminated.cpu().numpy(),
            truncated.cpu().numpy(),
            {},
        )

    def close(self) -> None:
        # Propagate so gym.wrappers.RecordVideo (when --video is set)
        # finalises the MP4 and the underlying ManagerBasedRLEnv tears down.
        self._env.close()


# ─── Section 6: main ────────────────────────────────────────────────────────

def main() -> None:
    # Resolve checkpoint path
    if args_cli.checkpoint is not None:
        ckpt_dir = Path(args_cli.checkpoint)
        if not ckpt_dir.is_dir():
            raise NotADirectoryError(f"Checkpoint path is not a directory: {ckpt_dir}")
    else:
        ckpt_dir = _find_latest_checkpoint(args_cli.task)

    print(f"[FlashSAC] Loading checkpoint: {ckpt_dir}")

    # Load agent config saved during training
    cfg = _load_config(ckpt_dir)
    # Override device in case we're on a different machine
    device_str: str = getattr(args_cli, "device", None) or "cuda:0"
    cfg = dataclasses.replace(cfg, device_type=device_str, use_amp=False)

    # Create environment
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # Make the viewport camera follow env 0's robot. Default ViewerCfg
    # uses origin_type="world" and a fixed eye at (7.5, 7.5, 7.5), so the
    # robot walks out of frame after a few seconds. With asset_root + a
    # 3 m chase offset, the camera stays locked on the robot's base for
    # the whole episode.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 1.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    # ── Mirror train.py's env overrides so eval matches training ────────
    # The trained policy never saw `randomize_actuator_gains`, was warm-started
    # on terrain level 0, and operated under one command per episode. Running
    # eval against the default rough_env_cfg exposes it to all of these at
    # once and the printed returns crash to near zero. This block makes
    # play.py reproduce the training distribution.
    env_cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.2),
        "yaw": (-3.14, 3.14),
    }
    env_cfg.events.randomize_reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-0.5, 0.5),
    }
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.commands.base_velocity.resampling_time_range = (20.0, 20.0)
    env_cfg.rewards.action_rate_l2.weight = -0.02
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 5.0
    env_cfg.rewards.track_ang_vel_z_exp.weight = 2.5

    # Match the actor's training-time obs shape. train.py writes an
    # `env_overrides.json` sidecar listing which obs terms it restored
    # to the policy group. For older checkpoints without the sidecar,
    # *infer* by reading the actor's expected input dim straight from
    # the checkpoint's BatchNorm running_mean shape — that way pre-fix
    # checkpoints (no lin_vel) and post-fix checkpoints (lin_vel only,
    # or lin_vel + height_scan) all just work.
    run_dir = ckpt_dir.parent
    overrides_path = run_dir / "env_overrides.json"
    if overrides_path.exists():
        env_overrides = json.loads(overrides_path.read_text())
        print(f"[FlashSAC] Loaded env_overrides.json: {env_overrides}")
    else:
        ckpt_actor_dim = _infer_actor_input_dim(ckpt_dir / "actor.pt")
        # Base policy obs (rough_env_cfg disables lin_vel and height_scan):
        #   base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
        #   + joint_pos(12 leg-only) + joint_vel(16) + last_actions(16) = 53
        # +3 if lin_vel restored → 56;  +160 (16×10) if height_scan → 213.
        # We pick the closest legal combination to the checkpoint's input.
        base = 53 + 4  # +4 fudge for any task-config drift; we still match exactly below
        candidates = {
            (False, False): 0,
            (True,  False): 3,
            (True,  True):  3 + 160,
            (False, True):  160,
        }
        if ckpt_actor_dim is None:
            print("[FlashSAC] Could not read checkpoint input dim — defaulting to lin_vel only.")
            env_overrides = {"restore_lin_vel": True, "restore_height_scan": False}
        else:
            best = min(candidates, key=lambda k: abs(ckpt_actor_dim - (base + candidates[k])))
            env_overrides = {"restore_lin_vel": best[0], "restore_height_scan": best[1]}
            print(f"[FlashSAC] Inferred from actor.pt (input_dim={ckpt_actor_dim}): {env_overrides}")

    if env_overrides.get("restore_lin_vel"):
        env_cfg.observations.policy.base_lin_vel = copy.deepcopy(
            env_cfg.observations.critic.base_lin_vel
        )
    if env_overrides.get("restore_height_scan"):
        env_cfg.observations.policy.height_scan = copy.deepcopy(
            env_cfg.observations.critic.height_scan
        )

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = Go2WIsaacEnvWrapper(raw_env, device=device_str)

    print(f"[FlashSAC] obs: {env.single_observation_space.shape}  act: {env.single_action_space.shape}")

    # Build agent and load weights
    obs, env_info = env.reset()
    agent = FlashSACAgent(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env_info=env_info,
        cfg=cfg,
    )
    agent.load(str(ckpt_dir))

    # Optional fixed velocity command override
    cmd_override: torch.Tensor | None = None
    if args_cli.cmd is not None:
        cmd_override = (
            torch.tensor(args_cli.cmd, device=device_str)
            .unsqueeze(0)
            .expand(env.num_envs, -1)
        )
        print(f"[FlashSAC] Command override: lin_x={args_cli.cmd[0]}, lin_y={args_cli.cmd[1]}, ang_z={args_cli.cmd[2]}")

    dt: float = env._isaac_env.step_dt

    # Play loop
    prev_transition: dict[str, Any] = {"next_observation": obs}
    ep_returns = np.zeros(env.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(env.num_envs, dtype=np.int32)
    completed = 0

    print(f"[FlashSAC] Playing {args_cli.num_episodes} episodes with {env.num_envs} envs...")

    # Camera-switch state. If --camera_switch_interval > 0 and we have >1
    # env, cycle the viewport's tracked env_index in real time so each robot
    # gets equal screen time.
    camera_env_idx = 0
    camera_last_switch = time.time()
    do_camera_switch = (args_cli.camera_switch_interval > 0.0) and (env.num_envs > 1)
    if do_camera_switch:
        print(f"[FlashSAC] Camera will cycle every {args_cli.camera_switch_interval:.0f}s "
              f"across {env.num_envs} robots.")

    while simulation_app.is_running() and completed < args_cli.num_episodes:
        step_start = time.time()

        if do_camera_switch and (time.time() - camera_last_switch) >= args_cli.camera_switch_interval:
            camera_env_idx = (camera_env_idx + 1) % env.num_envs
            try:
                env._isaac_env.viewport_camera_controller.set_view_env_index(camera_env_idx)
                print(f"  [camera] now following env {camera_env_idx}")
            except (AttributeError, RuntimeError):
                # ViewportCameraController API surface varies across IsaacLab
                # versions; if set_view_env_index isn't there, fall back to
                # mutating the cfg directly (the controller re-reads it).
                env._isaac_env.cfg.viewer.env_index = camera_env_idx
            camera_last_switch = time.time()

        if cmd_override is not None:
            env._isaac_env.command_manager.get_command("base_velocity")[:] = cmd_override

        with torch.inference_mode():
            actions: np.ndarray = agent.sample_actions(
                interaction_step=0, prev_transition=prev_transition, training=False
            )

        next_obs, rewards, terminated, truncated, _ = env.step(np.array(actions, dtype=np.float32))

        ep_returns += rewards
        ep_lengths += 1
        done_mask = terminated | truncated

        for i in range(env.num_envs):
            if done_mask[i]:
                completed += 1
                print(
                    f"  episode {completed:>3}  return={ep_returns[i]:>8.2f}  "
                    f"length={ep_lengths[i]:>5}"
                )
                ep_returns[i] = 0.0
                ep_lengths[i] = 0
                if completed >= args_cli.num_episodes:
                    break

        prev_transition = {"next_observation": next_obs}

        if args_cli.real_time:
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        # Isaac Sim's CARB layer occasionally leaves background threads alive
        # (USD asset loaders, render workers) so the process hangs after
        # main() returns. Force-exit to guarantee termination.
        os._exit(0)