"""Compare RSL-RL (PPO) and FlashSAC (SAC) checkpoints on shared metrics — Go2.

Runs each agent for `--num_episodes` episodes on the Go2 rough task and
collects per-episode means of:
  - linear / angular velocity tracking error
  - cost of transport (Σ|τ·ω| / m·g·v over moving steps)
  - joint power, mean speed, base height, orientation error
  - action-rate L2 (gait smoothness), episode return / length, survival rate

Both agents are evaluated in the SAME env (curriculum disabled, terrain levels
uniform across rows, randomizations off, one velocity command per episode),
using each agent's native obs / action conventions:
  - RSL-RL (PPO): proprioceptive 45-dim policy obs + asymmetric critic
  - FlashSAC (deployable): proprioceptive 45-dim actor + privileged critic

Defaults auto-discover the latest Go2 PPO run (unitree_go2_rough) and the
latest FlashSAC checkpoint under logs/flashsac/.
"""

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

# ─── Section 2: checkpoint auto-discovery ───────────────────────────────────

def _find_latest_ppo_ckpt() -> str | None:
    """Latest model_N.pt under robot_lab/logs/rsl_rl/unitree_go2_rough/."""
    root = Path(_robot_lab_root) / "logs" / "rsl_rl" / "unitree_go2_rough"
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        models = list(run.glob("model_*.pt"))
        if models:
            best = max(models, key=lambda p: int(p.stem.split("_")[1]))
            return str(best)
    return None


def _find_latest_flashsac_ckpt(task: str) -> str | None:
    logs_root = Path(_SCRIPT_DIR) / "logs" / "flashsac" / task
    if not logs_root.exists():
        return None
    candidates: list[Path] = []
    for run_dir in logs_root.iterdir():
        if not run_dir.is_dir():
            continue
        for ckpt_dir in run_dir.iterdir():
            if ckpt_dir.is_dir() and (ckpt_dir / "actor.pt").exists() and (
                ckpt_dir.name == "best" or ckpt_dir.name.startswith("step_") or ckpt_dir.name == "final"
            ):
                candidates.append(ckpt_dir)
    if not candidates:
        return None

    def _key(p: Path) -> tuple[int, int, int, float]:
        return (int(p.name == "best"), int(p.name == "final"),
                int(p.name.split("_")[1]) if p.name.startswith("step_") else 0,
                p.stat().st_mtime)
    return str(sorted(candidates, key=_key)[-1])


# ─── Section 3: CLI args + AppLauncher ──────────────────────────────────────
# FlashSAC (SAC) runs on the SAC-tuned task copy; the PPO baseline runs on the
# original task it was trained on (single-frame obs / original reward).
_DEFAULT_TASK = "RobotLab-Isaac-Velocity-Rough-Unitree-Go2-SAC-v0"
_DEFAULT_PPO_TASK = "RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0"

parser = argparse.ArgumentParser(description="Compare RSL-RL and FlashSAC on Go2.")
parser.add_argument("--task", type=str, default=_DEFAULT_TASK,
                    help="Task for the FlashSAC agent (SAC-tuned copy).")
parser.add_argument("--ppo_task", type=str, default=_DEFAULT_PPO_TASK,
                    help="Task for the RSL-RL/PPO baseline (the original Go2 task).")
parser.add_argument("--rsl_rl_ckpt", type=str, default=None,
                    help="Path to an RSL-RL .pt model file. Default: latest unitree_go2_rough run.")
parser.add_argument("--flashsac_ckpt", type=str, default=None,
                    help="FlashSAC checkpoint dir (contains actor.pt). Default: latest under logs/flashsac/.")
parser.add_argument("--agent", choices=["both", "rsl_rl", "flashsac"], default="both")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=200,
                    help="Per agent. Higher → tighter confidence intervals.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None,
                    help="Where to write the JSON results. "
                         "Default: <repo>/eval_results/compare_<timestamp>.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# headless by default (we never need a viewport for eval metrics)
if not getattr(args_cli, "headless", False):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Section 4: post-launcher imports ───────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry  # noqa: E402

if _LAB_MACHINE:
    import QuadLoco  # type: ignore  # noqa: F401
import go2_sac_env_cfg  # noqa: F401, E402  (registers SAC task + original Go2 ids)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig  # noqa: E402

GRAVITY = 9.81
# Joint order of the action term / proprioceptive obs (must match train.py).
JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def _per_joint_bounds(isaac_env, device):
    """Per-joint affine action mapping a = b + c·tanh, from soft joint limits."""
    robot = isaac_env.scene["robot"]
    jn = list(robot.data.joint_names)
    order = [jn.index(j) for j in JOINT_ORDER]
    default = robot.data.default_joint_pos[0, order].to(device)
    soft = robot.data.soft_joint_pos_limits[0, order].to(device)
    scale = torch.tensor([0.125 if "hip" in j else 0.25 for j in JOINT_ORDER],
                         dtype=torch.float32, device=device)
    a_min = (soft[:, 0] - default) / scale
    a_max = (soft[:, 1] - default) / scale
    return 0.5 * (a_max + a_min), 0.5 * (a_max - a_min)  # b, c


# ─── Section 5: shared env builder ──────────────────────────────────────────

def build_env_cfg(task: str, device: str, num_envs: int, seed: int,
                  policy_as_critic: bool = False):
    """Eval env: curriculum off, randomizations disabled, single 20-s command."""
    cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    cfg.seed = seed

    cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2),
        "yaw": (-3.14, 3.14),
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.5, 0.5),
    }
    cfg.events.randomize_actuator_gains = None
    if hasattr(cfg.events, "push_robot"):
        cfg.events.push_robot = None
    if hasattr(cfg.events, "randomize_apply_external_force_torque"):
        cfg.events.randomize_apply_external_force_torque = None
    cfg.observations.policy.enable_corruption = False

    cfg.commands.base_velocity.resampling_time_range = (20.0, 20.0)

    # Spread envs uniformly across terrain difficulty (no curriculum)
    cfg.scene.terrain.max_init_terrain_level = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
    if hasattr(cfg.curriculum, "terrain_levels"):
        cfg.curriculum.terrain_levels = None

    # Only needed if a PPO actor was trained on the full privileged obs.
    if policy_as_critic:
        cfg.observations.policy = copy.deepcopy(cfg.observations.critic)
    return cfg


# ─── Section 6: metric collector ────────────────────────────────────────────

class MetricCollector:
    """Accumulates per-step metrics on-device and finalises per episode."""

    def __init__(self, isaac_env: Any) -> None:
        self.env = isaac_env
        self.num_envs = isaac_env.num_envs
        self.robot = isaac_env.scene["robot"]
        self.device = isaac_env.device

        mass = self.robot.data.default_mass.to(self.device)
        self.mass = mass.sum(dim=-1)
        self.reset()

    def reset(self) -> None:
        N = self.num_envs
        d = self.device
        self.ep_return = torch.zeros(N, device=d)
        self.ep_length = torch.zeros(N, dtype=torch.long, device=d)
        self.lin_err_sum = torch.zeros(N, device=d)
        self.ang_err_sum = torch.zeros(N, device=d)
        self.power_sum = torch.zeros(N, device=d)
        self.speed_sum = torch.zeros(N, device=d)
        self.height_sum = torch.zeros(N, device=d)
        self.orient_sum = torch.zeros(N, device=d)
        self.action_rate_sum = torch.zeros(N, device=d)
        self.action_rate_steps = torch.zeros(N, dtype=torch.long, device=d)
        self.cot_num_sum = torch.zeros(N, device=d)
        self.cot_den_sum = torch.zeros(N, device=d)
        self.terrain_level_sum = torch.zeros(N, device=d)

        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []
        self.completed_lin_err: list[float] = []
        self.completed_ang_err: list[float] = []
        self.completed_power: list[float] = []
        self.completed_speed: list[float] = []
        self.completed_height: list[float] = []
        self.completed_orient: list[float] = []
        self.completed_action_rate: list[float] = []
        self.completed_cot: list[float] = []
        self.completed_terrain: list[float] = []
        self.completed_terminated: list[bool] = []

    def step(self, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        d = self.robot.data
        cmd = self.env.command_manager.get_command("base_velocity")

        lin_err = (cmd[:, :2] - d.root_lin_vel_b[:, :2]).norm(dim=-1)
        ang_err = (cmd[:, 2] - d.root_ang_vel_b[:, 2]).abs()
        power = (d.joint_vel * d.applied_torque).abs().sum(-1)
        speed = d.root_lin_vel_b[:, :2].norm(dim=-1)
        height = d.root_pos_w[:, 2]
        orient = (1.0 + d.projected_gravity_b[:, 2]).abs()

        am = self.env.action_manager
        action_rate = (am.action - am.prev_action).pow(2).sum(-1)

        # Per-env terrain level (the key adaptability KPI), if available.
        terrain = getattr(self.env.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None) if terrain is not None else None

        rew = rewards if rewards.dim() == 1 else rewards.squeeze(-1)
        self.ep_return += rew
        self.ep_length += 1
        self.lin_err_sum += lin_err
        self.ang_err_sum += ang_err
        self.power_sum += power
        self.speed_sum += speed
        self.height_sum += height
        self.orient_sum += orient
        if levels is not None:
            self.terrain_level_sum += levels.float()

        ar_mask = self.ep_length > 1
        self.action_rate_sum += torch.where(ar_mask, action_rate, torch.zeros_like(action_rate))
        self.action_rate_steps += ar_mask.long()

        moving = speed > 0.1
        self.cot_num_sum += torch.where(moving, power, torch.zeros_like(power))
        self.cot_den_sum += torch.where(moving, self.mass * GRAVITY * speed, torch.zeros_like(power))

        done = terminated | truncated
        if not done.any():
            return
        for idx in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            L = int(self.ep_length[idx].item())
            if L <= 0:
                continue
            self.completed_returns.append(float(self.ep_return[idx].item()))
            self.completed_lengths.append(L)
            self.completed_lin_err.append(float(self.lin_err_sum[idx].item() / L))
            self.completed_ang_err.append(float(self.ang_err_sum[idx].item() / L))
            self.completed_power.append(float(self.power_sum[idx].item() / L))
            self.completed_speed.append(float(self.speed_sum[idx].item() / L))
            self.completed_height.append(float(self.height_sum[idx].item() / L))
            self.completed_orient.append(float(self.orient_sum[idx].item() / L))
            self.completed_terrain.append(float(self.terrain_level_sum[idx].item() / L))
            ar_steps = int(self.action_rate_steps[idx].item())
            if ar_steps > 0:
                self.completed_action_rate.append(
                    float(self.action_rate_sum[idx].item() / ar_steps)
                )
            den = float(self.cot_den_sum[idx].item())
            if den > 1e-6:
                self.completed_cot.append(float(self.cot_num_sum[idx].item() / den))
            self.completed_terminated.append(bool(terminated[idx].item()))

            self.ep_return[idx] = 0.0
            self.ep_length[idx] = 0
            self.lin_err_sum[idx] = 0.0
            self.ang_err_sum[idx] = 0.0
            self.power_sum[idx] = 0.0
            self.speed_sum[idx] = 0.0
            self.height_sum[idx] = 0.0
            self.orient_sum[idx] = 0.0
            self.action_rate_sum[idx] = 0.0
            self.action_rate_steps[idx] = 0
            self.cot_num_sum[idx] = 0.0
            self.cot_den_sum[idx] = 0.0
            self.terrain_level_sum[idx] = 0.0

    def n_completed(self) -> int:
        return len(self.completed_returns)

    def summary(self, label: str) -> dict:
        def stat(xs: list[float]) -> dict:
            if not xs:
                return {"mean": float("nan"), "std": float("nan"), "n": 0}
            arr = np.asarray(xs, dtype=np.float64)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}

        n_eps = max(1, len(self.completed_terminated))
        survival = 1.0 - sum(self.completed_terminated) / n_eps

        return {
            "label": label,
            "n_episodes": self.n_completed(),
            "ep_return": stat(self.completed_returns),
            "ep_length": stat([float(x) for x in self.completed_lengths]),
            "mean_terrain_level": stat(self.completed_terrain),
            "lin_vel_tracking_error": stat(self.completed_lin_err),
            "ang_vel_tracking_error": stat(self.completed_ang_err),
            "joint_power_W": stat(self.completed_power),
            "mean_speed_mps": stat(self.completed_speed),
            "base_height_m": stat(self.completed_height),
            "orientation_error": stat(self.completed_orient),
            "action_rate_l2": stat(self.completed_action_rate),
            "cost_of_transport": stat(self.completed_cot),
            "survival_rate": survival,
        }


# ─── Section 7: RSL-RL evaluation ───────────────────────────────────────────

def _infer_ppo_actor_input_dim(ckpt_path: str) -> int | None:
    try:
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except (FileNotFoundError, RuntimeError):
        return None
    state = loaded.get("model_state_dict", loaded)
    w = state.get("actor.0.weight")
    if w is None:
        return None
    return int(w.shape[1])


def evaluate_rsl_rl(ckpt_path: str, num_envs: int, num_episodes: int,
                    seed: int, device: str, task: str) -> dict:
    print(f"\n[RSL-RL] checkpoint: {ckpt_path}", flush=True)

    actor_dim = _infer_ppo_actor_input_dim(ckpt_path)
    if actor_dim is None:
        print("[RSL-RL] could not read actor.0.weight — defaulting to policy_as_critic=False", flush=True)
        policy_as_critic = False
    elif actor_dim >= 200:
        print(f"[RSL-RL] actor input dim={actor_dim} → policy_as_critic=True (privileged obs)", flush=True)
        policy_as_critic = True
    else:
        print(f"[RSL-RL] actor input dim={actor_dim} → policy_as_critic=False (proprioceptive obs)", flush=True)
        policy_as_critic = False

    env_cfg = build_env_cfg(task, device, num_envs, seed, policy_as_critic=policy_as_critic)

    raw_env = gym.make(task, cfg=env_cfg, render_mode=None)
    isaac_env = raw_env.unwrapped

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    agent_cfg.seed = seed

    env = RslRlVecEnvWrapper(raw_env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(ckpt_path))
    policy = runner.get_inference_policy(device=device)

    raw_env.reset()
    collector = MetricCollector(isaac_env)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"[RSL-RL] obs={tuple(obs.shape)}  target={num_episodes} episodes "
          f"({num_envs} envs)", flush=True)

    step_count = 0
    last_log = 0
    while simulation_app.is_running() and collector.n_completed() < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            step_out = env.step(actions)
        obs, rew = step_out[0], step_out[1]
        if isinstance(obs, tuple):
            obs = obs[0]
        term = isaac_env.termination_manager.terminated.bool()
        tout = isaac_env.termination_manager.time_outs.bool()
        collector.step(rew, term, tout)

        step_count += 1
        if step_count % 200 == 0:
            print(f"  [heartbeat] sim_steps={step_count:5d}  "
                  f"episodes_done={collector.n_completed():4d}", flush=True)

        n_done = collector.n_completed()
        if n_done >= last_log + 25:
            recent = collector.completed_returns[-25:]
            recent_lin = collector.completed_lin_err[-25:]
            print(f"  episodes={n_done:4d}  ep_return(recent25)={np.mean(recent):.2f}  "
                  f"lin_err(recent25)={np.mean(recent_lin):.3f}", flush=True)
            last_log = n_done

    summary = collector.summary("RSL-RL (PPO)")
    env.close()
    return summary


# ─── Section 8: FlashSAC evaluation (deployable actor) ──────────────────────

def _default_flashsac_config() -> FlashSACConfig:
    return FlashSACConfig(
        seed=42, normalize_reward=True, normalized_G_max=5.0,
        asymmetric_observation=True, device_type="cuda",
        buffer_max_length=300_000, buffer_min_length=50_000,
        buffer_device_type="cuda", sample_batch_size=1024,
        learning_rate_init=3e-4, learning_rate_peak=3e-4, learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6, learning_rate_warmup_step=1,
        learning_rate_decay_rate=1.0, learning_rate_decay_step=1,
        actor_num_blocks=3, actor_hidden_dim=256, actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0, actor_noise_zeta_max=16, actor_update_period=2,
        critic_num_blocks=2, critic_hidden_dim=256, critic_num_bins=101,
        critic_min_v=-5.0, critic_max_v=5.0, critic_target_update_tau=0.01,
        temp_initial_value=0.01, temp_target_sigma=0.12, temp_target_entropy=0.0,
        gamma=0.99, n_step=3, use_compile=True, compile_mode="auto",
        use_amp=False, load_optimizer=False, load_reward_normalizer=False,
    )


def _build_full_obs(obs_dict, height_scan_start: int) -> np.ndarray:
    """Fully-blind layout: [proprio-history policy, base_lin_vel, height_scan]."""
    policy = obs_dict["policy"]
    critic = obs_dict["critic"]
    base_lin_vel = critic[:, 0:3]
    height_scan = critic[:, height_scan_start:]
    return torch.cat([policy, base_lin_vel, height_scan], dim=-1).cpu().numpy()


def evaluate_flashsac(ckpt_dir: str, num_envs: int, num_episodes: int,
                      seed: int, device: str, task: str) -> dict:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        raise NotADirectoryError(f"FlashSAC checkpoint must be a directory: {ckpt_path}")
    print(f"\n[FlashSAC] checkpoint: {ckpt_path}")

    run_dir = ckpt_path.parent
    cfg_json = run_dir / "config.json"
    if cfg_json.exists():
        flash_cfg = FlashSACConfig(**json.loads(cfg_json.read_text()))
    else:
        print("[FlashSAC] config.json missing — using fallback config")
        flash_cfg = _default_flashsac_config()
    flash_cfg = dataclasses.replace(flash_cfg, device_type=device, use_amp=False)

    # Fully-blind actor (proprio history, no base_lin_vel/height_scan); critic
    # sees the full obs incl base_lin_vel + height_scan. Matches train.py.
    obs_history = 1
    ov = run_dir / "env_overrides.json"
    if ov.exists():
        obs_history = int(json.loads(ov.read_text()).get("obs_history", 0) or 1)

    env_cfg = build_env_cfg(task, device, num_envs, seed, policy_as_critic=False)
    raw_env = gym.make(task, cfg=env_cfg, render_mode=None)
    isaac_env = raw_env.unwrapped

    policy_dim = int(isaac_env.single_observation_space["policy"].shape[-1])   # actor (proprio×H)
    critic_dim = int(isaac_env.single_observation_space["critic"].shape[-1])   # full single-frame (≈235)
    single_frame_proprio = policy_dim // max(1, obs_history)
    height_scan_start = single_frame_proprio + 3
    full_dim = policy_dim + 3 + (critic_dim - height_scan_start)
    act_shape = isaac_env.single_action_space.shape

    a_offset, a_scale = _per_joint_bounds(isaac_env, device)

    obs_space = gym.spaces.Box(low=0.0, high=0.0, shape=(full_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=act_shape, dtype=np.float32)

    obs_dict, _ = raw_env.reset()
    obs_np = _build_full_obs(obs_dict, height_scan_start)

    env_info = {
        "actor_observation_size": (policy_dim,),
        "asymmetric_obs": True,
    }
    agent = FlashSACAgent(
        observation_space=obs_space, action_space=act_space,
        env_info=env_info, cfg=flash_cfg,
    )
    agent.load(str(ckpt_path))

    collector = MetricCollector(isaac_env)
    prev_transition = {"next_observation": obs_np}

    print(f"[FlashSAC] full obs={full_dim}  actor obs={policy_dim}  act={act_shape}  "
          f"target={num_episodes} episodes ({num_envs} envs)", flush=True)

    step_count = 0
    last_log = 0
    while simulation_app.is_running() and collector.n_completed() < num_episodes:
        with torch.inference_mode():
            actions_np = agent.sample_actions(
                interaction_step=0, prev_transition=prev_transition, training=False,
            )
        torch_actions = torch.clamp(
            torch.as_tensor(np.asarray(actions_np, dtype=np.float32), device=device), -1.0, 1.0
        )
        torch_actions = a_offset + a_scale * torch_actions  # per-joint b + c·tanh
        obs_dict, rew, term, tout, _ = raw_env.step(torch_actions)
        prev_transition = {"next_observation": _build_full_obs(obs_dict, height_scan_start)}
        collector.step(rew, term.bool(), tout.bool())

        step_count += 1
        if step_count % 200 == 0:
            print(f"  [heartbeat] sim_steps={step_count:5d}  "
                  f"episodes_done={collector.n_completed():4d}", flush=True)

        n_done = collector.n_completed()
        if n_done >= last_log + 25:
            recent = collector.completed_returns[-25:]
            recent_lin = collector.completed_lin_err[-25:]
            print(f"  episodes={n_done:4d}  ep_return(recent25)={np.mean(recent):.2f}  "
                  f"lin_err(recent25)={np.mean(recent_lin):.3f}", flush=True)
            last_log = n_done

    summary = collector.summary("FlashSAC (SAC)")
    raw_env.close()
    return summary


# ─── Section 9: pretty-print + main ─────────────────────────────────────────

ROW_ORDER = [
    ("mean_terrain_level", "mean terrain level", "higher better — KPI"),
    ("lin_vel_tracking_error", "lin_vel tracking err [m/s]", "lower better"),
    ("ang_vel_tracking_error", "ang_vel tracking err [rad/s]", "lower better"),
    ("cost_of_transport", "cost of transport", "lower better"),
    ("joint_power_W", "joint power [W]", "lower better"),
    ("mean_speed_mps", "mean speed [m/s]", ""),
    ("base_height_m", "base height [m]", "≈0.33 nominal"),
    ("orientation_error", "orientation error", "lower better"),
    ("action_rate_l2", "action rate L2", "lower better"),
    ("ep_return", "ep return (canonical reward)", "higher better"),
    ("ep_length", "ep length [steps]", "higher better"),
]


def print_table(summaries: list[dict]) -> None:
    if not summaries:
        return
    label_w = 30
    col_w = 26
    print()
    header = "metric".ljust(label_w) + "".join(s["label"].ljust(col_w) for s in summaries)
    print(header)
    print("-" * len(header))
    for key, name, hint in ROW_ORDER:
        row = name.ljust(label_w)
        for s in summaries:
            v = s[key]
            if v["n"] == 0:
                cell = "—"
            else:
                cell = f"{v['mean']:>8.3f} ± {v['std']:<7.3f}"
            row += cell.ljust(col_w)
        if hint:
            row += f"   ({hint})"
        print(row)
    print()
    for s in summaries:
        print(f"  {s['label']}: survival={s['survival_rate']:.1%}  "
              f"n_episodes={s['n_episodes']}")


def main() -> None:
    device = getattr(args_cli, "device", None) or "cuda:0"

    rsl_rl_ckpt = args_cli.rsl_rl_ckpt or _find_latest_ppo_ckpt()
    flashsac_ckpt = args_cli.flashsac_ckpt or _find_latest_flashsac_ckpt(args_cli.task)

    summaries: list[dict] = []
    if args_cli.agent in ("both", "rsl_rl"):
        if not rsl_rl_ckpt:
            print("[RSL-RL] no checkpoint found/given — skipping PPO baseline.")
        else:
            summaries.append(evaluate_rsl_rl(
                rsl_rl_ckpt, args_cli.num_envs, args_cli.num_episodes,
                args_cli.seed, device, args_cli.ppo_task,
            ))
    if args_cli.agent in ("both", "flashsac"):
        if not flashsac_ckpt:
            print("[FlashSAC] no checkpoint found/given — skipping.")
        else:
            summaries.append(evaluate_flashsac(
                flashsac_ckpt, args_cli.num_envs, args_cli.num_episodes,
                args_cli.seed, device, args_cli.task,
            ))

    out_path = args_cli.output or os.path.join(
        _SCRIPT_DIR, "eval_results", f"compare_{int(time.time())}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": args_cli.task,
                "num_envs": args_cli.num_envs,
                "num_episodes_target": args_cli.num_episodes,
                "seed": args_cli.seed,
                "rsl_rl_ckpt": rsl_rl_ckpt,
                "flashsac_ckpt": flashsac_ckpt,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"\n[OK] saved → {out_path}")
    print_table(summaries)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()
        os._exit(0)
