"""Train regular Go2 (legged) rough-terrain locomotion with FlashSAC.

Deployable-actor variant. Unlike the Go2W "v7" pipeline (which restored
``base_lin_vel`` into the *policy* observation to break a tracking plateau,
making the actor non-deployable), this script keeps the actor strictly
proprioceptive and gives the privileged information only to the critic:

  actor  obs (45-dim, deployable): base_ang_vel, projected_gravity,
      velocity_commands, joint_pos, joint_vel, last_action
  critic obs (privileged):         actor obs + base_lin_vel + height_scan

This mirrors the asymmetric setup the PPO baseline used to reach mean terrain
level ~4 on RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0. FlashSAC stores the
full critic obs in the replay buffer, feeds the full obs to the critic, and
slices the first 45 dims for the actor (cfg.asymmetric_observation=True), so
the trained actor consumes exactly the 45-dim deployable observation.
"""

# ─── Section 1: stdlib + sys.path setup ─────────────────────────────────────
# All non-stdlib imports happen AFTER AppLauncher launches below.
import argparse
import collections
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

# Inject robot_lab (same pattern as go2w_flashsac/train.py)
_LAB_MACHINE = os.path.isdir("/home/roblab/quadruped_lab")
_robot_lab_root = os.environ.get("ROBOT_LAB_DIR", os.path.expanduser("~/robotics/robot_lab"))
if not _LAB_MACHINE:
    sys.path.insert(0, os.path.join(_robot_lab_root, "source", "robot_lab"))

# Inject FlashSAC (cloned as a sibling subdirectory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "FlashSAC"))

from isaaclab.app import AppLauncher  # noqa: E402  (must be imported before Isaac Sim launches)

# ─── Section 2: CLI args + AppLauncher (MUST happen before any Isaac import) ─
parser = argparse.ArgumentParser(description="Train Go2 rough terrain with FlashSAC (deployable actor).")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-SAC-v0",
    help="IsaacLab gym task ID (SAC-tuned copy; see go2_sac_env_cfg.py).",
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--total_steps", type=int, default=50_000_000, help="Total environment steps.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--wandb_name", type=str, default="go2_flashsac_v1", help="WandB run name.")
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

# Task registration — importing go2_sac_env_cfg registers the SAC task id
# (and, via its robot_lab import, the original Go2 ids too).
if _LAB_MACHINE:
    try:
        import QuadLoco  # type: ignore  # noqa: F401  (legacy lab pkg; optional)
    except Exception:
        pass
import go2_sac_env_cfg  # noqa: F401, E402  (registers RobotLab-...-Go2-SAC-v0)
try:
    import fablequadruped.tasks  # noqa: F401, E402  (registers Fable-Go2-* for --task Fable-...)
except Exception as _e:  # noqa: BLE001
    print(f"[train] fablequadruped.tasks not importable ({_e}); Fable tasks unavailable.")
try:
    import quadruped_lab.tasks  # noqa: F401  (registers QuadrupedLab-...-Fable-v0 on the lab PC)
except Exception as _e:  # noqa: BLE001
    print(f"[train] quadruped_lab.tasks not importable ({_e}); QuadrupedLab tasks unavailable.")

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig  # noqa: E402

# ─── Section 4: RTX 4070 laptop torch settings ───────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ─── Section 5: Go2 IsaacLab environment wrapper (RSL-RL-SAC faithful) ───────
# Joint order of the action term / proprioceptive obs (rough_env_cfg.joint_names).
JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class Go2IsaacEnvWrapper:
    """
    Wraps a ManagerBasedRLEnv to the gymnasium VectorEnv-like interface FlashSAC
    expects, implementing the RSL-RL-SAC fixes:

    * **Blind asymmetric obs** — the actor consumes the policy group
      (base_lin_vel + proprioception, ≈48-dim, NO height_scan), which is a strict
      prefix of the privileged critic obs (policy + height_scan, ≈235-dim). The
      buffer/critic see the full obs; FlashSAC slices the first 48 dims for the
      actor (asymmetric_observation=True). Single frame, no history.
    * **#1 Per-joint asymmetric action bounds** — the actor's tanh∈[−1,1] is
      affine-mapped per joint to ``a = b_j + c_j·tanh`` where ``[a_min,a_max]`` are
      derived from the robot's soft joint limits (eq 5). Replaces a global scalar.
    * **#2 Timeout bootstrap** — `_reset_idx` is monkeypatched to stash the
      *pre-reset* observation, exposed as ``infos["final_obs"]`` so the buffer's
      ``next_observation`` for timeout transitions is the true terminal state.
    """

    def __init__(self, env: Any, device: str, obs_history: int = 1) -> None:
        self._env = env
        self._isaac_env = env.unwrapped
        self.device = device
        self.num_envs: int = self._isaac_env.num_envs
        self.max_episode_steps: int = self._isaac_env.max_episode_length

        obs_spaces = self._isaac_env.single_observation_space
        policy_shape = obs_spaces["policy"].shape    # actor: proprio × history (e.g. 45×5=225)
        action_shape = self._isaac_env.single_action_space.shape
        self._actor_obs_dim = int(policy_shape[-1])          # 225 (proprio history)
        frames = max(1, obs_history)

        # Derive the critic slicing (base_lin_vel / height_scan located BY NAME)
        # + per-joint action scale from the env's own metadata, asserting loudly
        # on any mismatch — so running on Fable vs robot_lab can't silently slice
        # the wrong critic columns or use the wrong action scale. See env_wiring.py.
        from env_wiring import self_check
        self._wiring = self_check(self._isaac_env, self._actor_obs_dim, frames, JOINT_ORDER)
        self._blv_start, self._blv_dim = self._wiring["blv"]
        self._height_scan = self._wiring["height_scan"]   # (start, dim) or None (flat)
        self._full_obs_dim = self._wiring["full_obs_dim"]

        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=(self._full_obs_dim,), dtype=np.float32
        )
        self._actor_observation_shape = (self._actor_obs_dim,)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        # The actor natively outputs tanh∈[−1,1]; per-joint scaling is applied in
        # step(). So the agent's action space is the tanh range.
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=action_shape, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self._compute_per_joint_bounds()
        self._install_pre_reset_hook()

    def _build_full_obs(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """[proprio history, base_lin_vel, height_scan] → full FlashSAC obs. The
        actor consumes the proprio history; base_lin_vel + height_scan are pulled
        from the critic group at their metadata-derived offsets (env_wiring)."""
        policy = obs_dict["policy"]
        critic = obs_dict["critic"]
        base_lin_vel = critic[:, self._blv_start:self._blv_start + self._blv_dim]
        if self._height_scan is not None:
            hs, hd = self._height_scan
            return torch.cat([policy, base_lin_vel, critic[:, hs:hs + hd]], dim=-1)
        return torch.cat([policy, base_lin_vel], dim=-1)

    # ── #1 per-joint asymmetric action bounds (from soft joint limits) ──────
    def _compute_per_joint_bounds(self) -> None:
        robot = self._isaac_env.scene["robot"]
        jn = list(robot.data.joint_names)
        order = [jn.index(j) for j in JOINT_ORDER]
        default = robot.data.default_joint_pos[0, order].to(self.device)          # [12]
        soft = robot.data.soft_joint_pos_limits[0, order].to(self.device)         # [12,2]
        scale = torch.tensor(
            self._wiring["action_scale"],  # derived from the env action term, not hardcoded
            dtype=torch.float32, device=self.device,
        )
        a_min = (soft[:, 0] - default) / scale   # tanh=-1 → q_soft_min
        a_max = (soft[:, 1] - default) / scale   # tanh=+1 → q_soft_max
        self._a_offset = 0.5 * (a_max + a_min)   # b_j
        self._a_scale = 0.5 * (a_max - a_min)    # c_j
        self.action_bounds = {
            "a_min": a_min.cpu().tolist(), "a_max": a_max.cpu().tolist(),
            "joint_order": JOINT_ORDER,
        }

    # ── #2 capture the pre-reset (full) observation for timeout bootstrap ───
    def _install_pre_reset_hook(self) -> None:
        self._pre_reset_full: np.ndarray | None = None
        _orig_reset_idx = self._isaac_env._reset_idx

        def _patched_reset_idx(env_ids):  # type: ignore[no-untyped-def]
            # Compute obs at the (post-physics) state BEFORE the reset overwrites
            # it, so terminating/timing-out envs keep their true final obs.
            try:
                obs = self._isaac_env.observation_manager.compute()
                self._pre_reset_full = self._build_full_obs(obs).cpu().numpy()
            except Exception:  # noqa: BLE001
                self._pre_reset_full = None
            return _orig_reset_idx(env_ids)

        self._isaac_env._reset_idx = _patched_reset_idx

    def _map_actions(self, actions: np.ndarray) -> torch.Tensor:
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        a = torch.clamp(a, -1.0, 1.0)
        return self._a_offset + self._a_scale * a  # b + c·tanh, per joint

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, _ = self._isaac_env.reset(seed=seed)
        if random_start_init:
            self._isaac_env.episode_length_buf = torch.randint_like(
                self._isaac_env.episode_length_buf, high=int(self.max_episode_steps),
            )
        env_info = {
            "actor_observation_size": self._actor_observation_shape,
            "asymmetric_obs": True,
        }
        return self._build_full_obs(obs_dict).cpu().numpy(), env_info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs_dict, rew, terminated, truncated, _ = self._isaac_env.step(self._map_actions(actions))

        obs_np = self._build_full_obs(obs_dict).cpu().numpy()
        term_np = terminated.cpu().numpy()
        trunc_np = truncated.cpu().numpy()

        # #2: for envs that reset this step, the returned obs is post-reset; use
        # the stashed pre-reset full obs as the true final observation.
        final_obs = obs_np.copy()
        done_mask = term_np | trunc_np
        if done_mask.any() and self._pre_reset_full is not None:
            final_obs[done_mask] = self._pre_reset_full[done_mask]

        infos: dict[str, Any] = {"final_obs": final_obs, "time_outs": trunc_np}
        return obs_np, rew.cpu().numpy(), term_np, trunc_np, infos

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
            _SCRIPT_DIR, "logs", "flashsac", args_cli.task, f"{WANDB_RUN_NAME}_{run_timestamp}"
        )
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir, "command.txt").write_text(" ".join(sys.argv))
    print(f"[FlashSAC] Logging to: {log_dir}")

    # WandB
    wandb.init(
        project="flashsac_go2",
        name=WANDB_RUN_NAME,
        config={
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "total_steps": args_cli.total_steps,
            "seed": seed,
            "algorithm": "FlashSAC-RSL (full obs)",
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    # Environment. The SAC task copy (go2_sac_env_cfg.UnitreeGo2RoughSACEnvCfg)
    # owns the env: robot_lab's unmodified Go2 reward + fully-blind proprio actor
    # obs (history-stacked; NO base_lin_vel/height_scan) + curriculum-from-0 +
    # upright spawn. The wrapper adds per-joint action bounds + the timeout
    # pre-reset obs and builds the privileged critic obs (+ base_lin_vel +
    # height_scan); the actor prefix is the proprio history.
    device_str: str = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.seed = seed
    obs_history = int(getattr(env_cfg.observations.policy, "history_length", 0) or 1)

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = Go2IsaacEnvWrapper(raw_env, device=device_str, obs_history=obs_history)

    print(
        f"[FlashSAC] full obs: {env.single_observation_space.shape}  "
        f"actor obs: {env._actor_observation_shape} (proprio history={obs_history})  "
        f"act: {env.single_action_space.shape}  num_envs: {env.num_envs}\n"
        f"[FlashSAC] per-joint action a_min: "
        f"{[round(x,2) for x in env.action_bounds['a_min']]}\n"
        f"[FlashSAC] per-joint action a_max: "
        f"{[round(x,2) for x in env.action_bounds['a_max']]}"
    )

    # Sidecar for play.py / evaluate.py / export_policy.py.
    Path(log_dir, "env_overrides.json").write_text(json.dumps({
        "task": args_cli.task,
        "blind_actor": True,          # proprioception only: NO base_lin_vel, NO height_scan
        "asymmetric_critic": True,    # critic sees base_lin_vel + height_scan (privileged)
        "actor_obs_dim": env._actor_obs_dim,
        "obs_history": obs_history,   # proprio frames stacked
        "action_bounds": env.action_bounds,   # per-joint a_min/a_max + joint order
    }, indent=2))

    # FlashSAC config — RSL-RL-SAC hyperparameters (paper Table C9), adapted to
    # our 8 GB GPU (smaller batch, CPU replay buffer).
    num_envs = env.num_envs
    updates_per_step = 2  # UTD ≈ 2*1024/256 ≈ 8, matches the paper
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
        # Asymmetric: critic sees the full 235-dim obs (incl height_scan); the
        # blind actor uses the first ~48 dims (base_lin_vel + proprio).
        asymmetric_observation=True,
        device_type=device_str,
        # Large replay buffer on CPU (paper uses 5M on a 24 GB GPU; we hold ~2M
        # on host RAM — small buffer + UTD≈8 was a prime collapse driver).
        buffer_max_length=2_000_000,
        buffer_min_length=50_000,
        buffer_device_type="cpu",
        sample_batch_size=1024,
        # Learning rate — paper 2e-4, ~constant.
        learning_rate_init=2e-4,
        learning_rate_peak=2e-4,
        learning_rate_end=2e-4,
        learning_rate_warmup_rate=warmup_rate,
        learning_rate_warmup_step=warmup_step,
        learning_rate_decay_rate=decay_rate,
        learning_rate_decay_step=decay_step,
        actor_num_blocks=3,
        actor_hidden_dim=512,  # v7: widen (was 256) to raise the skill ceiling
        actor_bc_alpha=0.0,
        actor_smooth_beta=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=16,
        actor_update_period=1,          # paper: policy update frequency 1
        # Critic
        critic_num_blocks=2,
        critic_hidden_dim=512,  # v7: widen (was 256)
        critic_num_bins=101,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.003,  # paper τ
        # Temperature — target entropy ≈ -0.167·12 ≈ -2.0 (paper target-entropy
        # scale 0.167). temp_target_sigma=0.205 → 0.5·12·ln(2πe σ²) ≈ -2.0.
        # Higher (less negative) than our prior -8.4 → sustains exploration.
        temp_initial_value=0.001,
        # σ=0.205 → target entropy ≈ −2 (the v5/v6 level that learned to walk).
        # v7/v8's σ=0.25 (target +0.39) was too stochastic — the policy jittered
        # in place and never covered distance, so terrain stayed ~0. Reverted.
        temp_target_sigma=0.205,
        temp_target_entropy=0.0,  # overridden by FlashSACAgent.__init__
        temp_learning_rate=2e-5,  # decoupled, small (paper)
        # Controlled init (eq 7-8): start at default pose, σ₀=0.15.
        init_sigma=0.15,
        # RL
        gamma=0.97,        # paper
        n_step=5,          # paper multi-step horizon
        # Performance
        use_compile=True,
        compile_mode="auto",
        use_amp=True,
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

    BEST_MIN_EPISODES = 50  # don't save until the rolling mean is stable
    best_ep_rew_mean = float("-inf")
    # Also track the best checkpoint by the actual KPI (mean terrain level), so a
    # later collapse doesn't lose the peak policy. Updated from the logged value.
    best_terrain_level = float("-inf")
    latest_terrain_level = 0.0

    # Sliding window over the last 100 completed episodes (SB3 / RSL-RL
    # convention) so the displayed rollout/ep_rew_mean reflects sustained
    # policy quality.
    ep_rew_buf = np.zeros(num_envs, dtype=np.float32)
    ep_len_buf = np.zeros(num_envs, dtype=np.int32)
    completed_ep_rews: collections.deque[float] = collections.deque(maxlen=100)
    completed_ep_lens: collections.deque[int] = collections.deque(maxlen=100)

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

                # Terrain-level KPI: mean/max per-env terrain level. This is the
                # success metric (target mean >= 4), so log it explicitly.
                try:
                    terrain = env._isaac_env.scene.terrain
                    levels = getattr(terrain, "terrain_levels", None)
                    if levels is not None:
                        latest_terrain_level = float(levels.float().mean().item())
                        log_dict["env/mean_terrain_level"] = latest_terrain_level
                        log_dict["env/max_terrain_level"] = float(levels.float().max().item())
                except (AttributeError, KeyError):
                    pass

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

                extra_str = ""
                if "rollout/ep_rew_mean" in log_dict:
                    extra_str += f"  ep_rew {log_dict['rollout/ep_rew_mean']:.1f}"
                if "env/mean_terrain_level" in log_dict:
                    extra_str += f"  terrain {log_dict['env/mean_terrain_level']:.2f}"
                print(
                    f"[FlashSAC] step {env_step:>10,}  "
                    f"fps {log_dict.get('fps', 0):.0f}  "
                    f"updates {update_step:>8,}  "
                    f"elapsed {elapsed/60:.1f}min"
                    f"{extra_str}"
                )

            # ── Checkpointing ────────────────────────────────────────────────
            if interaction_step - last_ckpt_step >= CHECKPOINT_EVERY:
                last_ckpt_step = interaction_step
                ckpt_path = os.path.join(log_dir, f"step_{env_step}")
                agent.save(ckpt_path)

                # Best-policy save: rolling ep_rew_mean over the last 100
                # completed episodes. Only valid once past warmup with a
                # stable sample.
                if (
                    agent.can_start_training()
                    and len(completed_ep_rews) >= BEST_MIN_EPISODES
                ):
                    rolling = float(np.mean(completed_ep_rews))
                    if rolling > best_ep_rew_mean:
                        best_ep_rew_mean = rolling
                        best_path = os.path.join(log_dir, "best")
                        agent.save(best_path)
                        Path(best_path, "best_metadata.json").write_text(
                            json.dumps({
                                "env_step": int(env_step),
                                "interaction_step": int(interaction_step),
                                "ep_rew_mean": rolling,
                                "ep_len_mean": (
                                    float(np.mean(completed_ep_lens))
                                    if completed_ep_lens else 0.0
                                ),
                                "n_episodes_in_window": len(completed_ep_rews),
                            }, indent=2)
                        )
                        print(f"[FlashSAC] new best ep_rew_mean = {rolling:.2f} "
                              f"(window={len(completed_ep_rews)} eps) → saved to best/")

                # Best-by-terrain save (the actual KPI). Preserves the peak
                # policy even if training later collapses (terrain → 0).
                if (
                    agent.can_start_training()
                    and len(completed_ep_rews) >= BEST_MIN_EPISODES
                    and latest_terrain_level > best_terrain_level
                ):
                    best_terrain_level = latest_terrain_level
                    bt_path = os.path.join(log_dir, "best_terrain")
                    agent.save(bt_path)
                    Path(bt_path, "best_metadata.json").write_text(
                        json.dumps({
                            "env_step": int(env_step),
                            "interaction_step": int(interaction_step),
                            "mean_terrain_level": best_terrain_level,
                            "ep_rew_mean": (
                                float(np.mean(completed_ep_rews)) if completed_ep_rews else 0.0
                            ),
                        }, indent=2)
                    )
                    print(f"[FlashSAC] new best mean_terrain_level = {best_terrain_level:.3f} "
                          f"→ saved to best_terrain/")

    except KeyboardInterrupt:
        print("\n[FlashSAC] Training interrupted.")

    agent.save(os.path.join(log_dir, "final"))
    print(f"[FlashSAC] Total training time: {(time.time() - start_time) / 60:.1f} min")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
