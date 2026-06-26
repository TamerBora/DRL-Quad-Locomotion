"""Play a trained FlashSAC checkpoint on the regular Go2 task (deployable actor)."""

# ─── Section 1: stdlib + sys.path setup ─────────────────────────────────────
import argparse
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
parser = argparse.ArgumentParser(description="Play a trained FlashSAC Go2 checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-SAC-v0",
    help="IsaacLab gym task ID (SAC-tuned copy).",
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
import go2_sac_env_cfg  # noqa: F401, E402  (registers RobotLab-...-Go2-SAC-v0)
try:
    import fablequadruped.tasks  # noqa: F401, E402  (registers Fable-Go2-* for --task Fable-...)
except Exception as _e:  # noqa: BLE001
    print(f"[play] fablequadruped.tasks not importable ({_e}); Fable tasks unavailable.")

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig  # noqa: E402

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Joint order of the action term / proprioceptive obs (must match train.py).
JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


# ─── Section 4: checkpoint discovery ────────────────────────────────────────

def _find_latest_checkpoint(task: str) -> Path:
    """Return the most recent best/step_N/final dir under logs/flashsac/<task>/."""
    logs_root = Path(_SCRIPT_DIR) / "logs" / "flashsac" / task
    if not logs_root.exists():
        raise FileNotFoundError(
            f"No checkpoint found under {logs_root}.\n"
            "Run train.py first or pass --checkpoint explicitly."
        )

    candidates: list[Path] = []
    for run_dir in sorted(logs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for ckpt_dir in run_dir.iterdir():
            if ckpt_dir.is_dir() and (
                ckpt_dir.name == "best"
                or ckpt_dir.name.startswith("step_")
                or ckpt_dir.name == "final"
            ):
                if (ckpt_dir / "actor.pt").exists():
                    candidates.append(ckpt_dir)

    if not candidates:
        raise FileNotFoundError(
            f"Found run dirs under {logs_root} but no checkpoint subdirectories with actor.pt."
        )

    def _sort_key(p: Path) -> tuple[int, int, int, float]:
        is_best = int(p.name == "best")
        is_final = int(p.name == "final")
        step_num = int(p.name.split("_")[1]) if p.name.startswith("step_") else 0
        return (is_best, is_final, step_num, p.stat().st_mtime)

    latest = sorted(candidates, key=_sort_key)[-1]
    return latest


def _default_config() -> FlashSACConfig:
    """Fallback config matching train.py's deployable-actor defaults."""
    return FlashSACConfig(
        seed=42,
        normalize_reward=True,
        normalized_G_max=5.0,
        asymmetric_observation=True,
        device_type="cuda",
        buffer_max_length=300_000,
        buffer_min_length=50_000,
        buffer_device_type="cuda",
        sample_batch_size=1024,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=1,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=1,
        actor_num_blocks=3,
        actor_hidden_dim=256,
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
        temp_target_sigma=0.12,
        temp_target_entropy=0.0,
        gamma=0.99,
        n_step=3,
        use_compile=True,
        compile_mode="auto",
        use_amp=False,
        load_optimizer=False,
        load_reward_normalizer=False,
    )


def _load_config(ckpt_dir: Path) -> FlashSACConfig:
    """Load FlashSACConfig from config.json, falling back to defaults."""
    run_dir = ckpt_dir.parent if ckpt_dir.name in ("final", "best") or ckpt_dir.name.startswith("step_") else ckpt_dir
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"[FlashSAC] config.json not found at {config_path} — using default architecture config.")
        return _default_config()
    raw = json.loads(config_path.read_text())
    return FlashSACConfig(**raw)


# ─── Section 5: env wrapper (deployable-actor, mirrors train.py) ────────────

class Go2IsaacEnvWrapper:
    """Fully-blind asymmetric obs (actor = proprio history prefix; critic = full
    incl base_lin_vel + height_scan) + per-joint action bounds — matches train.py
    so the loaded actor's prefix and action mapping are identical."""

    def __init__(self, env: Any, device: str, obs_history: int = 1) -> None:
        self._env = env
        self._isaac_env = env.unwrapped
        self.device = device
        self.num_envs: int = self._isaac_env.num_envs
        self.max_episode_steps: int = self._isaac_env.max_episode_length

        obs_spaces = self._isaac_env.single_observation_space
        self._actor_obs_dim = int(obs_spaces["policy"].shape[-1])   # proprio × history
        frames = max(1, obs_history)
        # Metadata-derived critic slicing + action scale (see env_wiring.py).
        from env_wiring import self_check
        self._wiring = self_check(self._isaac_env, self._actor_obs_dim, frames, JOINT_ORDER)
        self._blv_start, self._blv_dim = self._wiring["blv"]
        self._height_scan = self._wiring["height_scan"]
        self._full_obs_dim = self._wiring["full_obs_dim"]
        action_shape = self._isaac_env.single_action_space.shape

        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=(self._full_obs_dim,), dtype=np.float32
        )
        self._actor_observation_shape = (self._actor_obs_dim,)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=action_shape, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._compute_per_joint_bounds()

    def _compute_per_joint_bounds(self) -> None:
        robot = self._isaac_env.scene["robot"]
        jn = list(robot.data.joint_names)
        order = [jn.index(j) for j in JOINT_ORDER]
        default = robot.data.default_joint_pos[0, order].to(self.device)
        soft = robot.data.soft_joint_pos_limits[0, order].to(self.device)
        scale = torch.tensor(
            self._wiring["action_scale"],  # derived from the env action term, not hardcoded
            dtype=torch.float32, device=self.device,
        )
        a_min = (soft[:, 0] - default) / scale
        a_max = (soft[:, 1] - default) / scale
        self._a_offset = 0.5 * (a_max + a_min)
        self._a_scale = 0.5 * (a_max - a_min)

    def _build_full_obs(self, obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
        policy = obs_dict["policy"]
        critic = obs_dict["critic"]
        base_lin_vel = critic[:, self._blv_start:self._blv_start + self._blv_dim]
        if self._height_scan is not None:
            hs, hd = self._height_scan
            full = torch.cat([policy, base_lin_vel, critic[:, hs:hs + hd]], dim=-1)
        else:
            full = torch.cat([policy, base_lin_vel], dim=-1)
        return full.cpu().numpy()

    def _map_actions(self, actions: np.ndarray) -> torch.Tensor:
        a = torch.clamp(torch.as_tensor(actions, dtype=torch.float32, device=self.device), -1.0, 1.0)
        return self._a_offset + self._a_scale * a

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, _ = self._env.reset()
        env_info: dict[str, Any] = {
            "actor_observation_size": self._actor_observation_shape,
            "asymmetric_obs": True,
        }
        return self._build_full_obs(obs_dict), env_info

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs_dict, rew, terminated, truncated, _ = self._env.step(self._map_actions(actions))
        return (
            self._build_full_obs(obs_dict),
            rew.cpu().numpy(),
            terminated.cpu().numpy(),
            truncated.cpu().numpy(),
            {},
        )

    def close(self) -> None:
        self._env.close()


# ─── Section 6: main ────────────────────────────────────────────────────────

def main() -> None:
    if args_cli.checkpoint is not None:
        ckpt_dir = Path(args_cli.checkpoint)
        if not ckpt_dir.is_dir():
            raise NotADirectoryError(f"Checkpoint path is not a directory: {ckpt_dir}")
    else:
        ckpt_dir = _find_latest_checkpoint(args_cli.task)

    print(f"[FlashSAC] Loading checkpoint: {ckpt_dir}")

    cfg = _load_config(ckpt_dir)
    device_str: str = getattr(args_cli, "device", None) or "cuda:0"
    cfg = dataclasses.replace(cfg, device_type=device_str, use_amp=False)

    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # Make the viewport camera follow env 0's robot.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 1.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    # The SAC task cfg (go2_sac_env_cfg) defines the env (blind actor obs,
    # robot_lab reward, curriculum start, upright spawn), so playback matches
    # training automatically. The wrapper recomputes the per-joint action bounds
    # from the same robot limits.
    run_dir = ckpt_dir.parent
    overrides_path = run_dir / "env_overrides.json"
    if overrides_path.exists():
        print(f"[FlashSAC] Loaded env_overrides.json: {json.loads(overrides_path.read_text())}")

    obs_history = int(getattr(env_cfg.observations.policy, "history_length", 0) or 1)
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = Go2IsaacEnvWrapper(raw_env, device=device_str, obs_history=obs_history)

    print(f"[FlashSAC] full obs: {env.single_observation_space.shape}  "
          f"actor obs: {env._actor_observation_shape}  act: {env.single_action_space.shape}")

    obs, env_info = env.reset()
    agent = FlashSACAgent(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env_info=env_info,
        cfg=cfg,
    )
    agent.load(str(ckpt_dir))

    cmd_override: torch.Tensor | None = None
    if args_cli.cmd is not None:
        cmd_override = (
            torch.tensor(args_cli.cmd, device=device_str)
            .unsqueeze(0)
            .expand(env.num_envs, -1)
        )
        print(f"[FlashSAC] Command override: lin_x={args_cli.cmd[0]}, lin_y={args_cli.cmd[1]}, ang_z={args_cli.cmd[2]}")

    dt: float = env._isaac_env.step_dt

    prev_transition: dict[str, Any] = {"next_observation": obs}
    ep_returns = np.zeros(env.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(env.num_envs, dtype=np.int32)
    completed = 0

    print(f"[FlashSAC] Playing {args_cli.num_episodes} episodes with {env.num_envs} envs...")

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
        os._exit(0)
