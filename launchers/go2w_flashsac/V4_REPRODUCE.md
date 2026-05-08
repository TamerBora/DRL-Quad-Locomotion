# Reproducing FlashSAC v4

This is the snapshot of the configuration that produced the best SAC results to date on the Go2W rough-terrain task: **terrain levels reaching 5.5**, **lin_vel_tracking_error 0.4**, robot upright (`base_height` 0.35–0.40), gait smooth (`action_rate_l2 ≈ 0`).

WandB run name: `flashsac_full_v4`. Checkpoint: 20 M env-steps.

---

## What v4 is

Configuration on top of vanilla FlashSAC + the rough-terrain task in `robot_lab`:

### FlashSAC core fixes (4 patches in `flashsac_patches/`)

| file | change | why |
|---|---|---|
| `layer.py` | `NormalTanhPolicy`: `log_std_min: -10 → -5`, `log_std_max: 2.0 → 0.0` | std capped at 1.0 stops the actor from saturating tanh into bang-bang outputs |
| `update.py` | `update_temperature`: switched from `α·(H−H̄)` to `log_α·(H−H̄)` (direct `log_temp` access) | with the original loss, gradient on `log α` is `α·Δ` — 100× too small at α₀=0.01, freezing α |
| `update.py` | `update_actor`: added `smooth_beta` param + actor-side action-rate term `β·Σ_dim(π(s')−π(s))²`; gradient clipping (max_norm 1.0 actor, 10.0 critic) | tames jitter; clipping catches FP16 overflow under AMP |
| `agent.py` | `FlashSACConfig`: added `actor_smooth_beta: float` field, threaded into `_update_networks → update_actor` | exposes the smoothness coefficient |

### Train-script overrides (`train.py`, in the env-overrides block)

- `pose_range`: yaw-only (no roll/pitch ±π) — robot spawns upright
- `velocity_range`: zero roll/pitch angular velocity — no spawn-time tipping
- `randomize_actuator_gains = None` — multimodal Q-landscape was killing the single critic
- `action_rate_l2.weight = -0.02` — 2× the baseline penalty
- `track_lin_vel_xy_exp.weight = 5.0`, `track_ang_vel_z_exp.weight = 2.5` — steeper tracking gradient
- `scene.terrain.max_init_terrain_level = 0` — all envs spawn at easiest level (curriculum starts properly)
- `commands.base_velocity.resampling_time_range = (20.0, 20.0)` — one command per 20-s episode (no mid-flight reversal that demotes the curriculum)
- `observations.policy.base_lin_vel = critic.base_lin_vel` — restore lin_vel to the policy obs (the structural unlock; PPO had it via asymmetric critic)

### Agent config (FlashSAC)

- `actor_num_blocks = 2`, `actor_hidden_dim = 128`
- `critic_num_blocks = 2`, `critic_hidden_dim = 256`
- `actor_smooth_beta = 0.0` (env-side action_rate handles smoothness)
- `temp_initial_value = 0.01`, `temp_target_sigma = 0.12`
- `buffer_min_length = 50_000`, `buffer_max_length = 500_000`
- `sample_batch_size = 1024`, `updates_per_step = 2`
- `gamma = 0.99`, `n_step = 3`, `critic_target_update_tau = 0.01`

---

## Reproducing from a fresh clone

### 1. Clone this repo

```bash
git clone https://github.com/TamerBora/DRL-Quad-Locomotion.git
cd DRL-Quad-Locomotion
```

### 2. Clone upstream FlashSAC and apply the v4 patches

```bash
cd go2w_flashsac
git clone https://github.com/Holiday-Robot/FlashSAC.git
cd FlashSAC
for p in ../flashsac_patches/*.patch; do
    git apply "$p"
done
cd ..
```

### 3. Install dependencies

Follow [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/) (Isaac Sim 4.5+, Python 3.11), then:

```bash
pip install -r requirements.txt
```

### 4. Train v4 (≈ 4 h on RTX 4070 8 GB)

```bash
python go2w_flashsac/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --num_envs 256 --total_steps 50_000_000 --seed 42 \
  --wandb_name flashsac_full_v4 --headless
```

### 5. Visualize a checkpoint

```bash
python go2w_flashsac/create_video.py \
  go2w_flashsac/logs/flashsac/RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0/<run-dir>/<step-dir> \
  --num_envs 3 --duration 60
```

The MP4 lands in `<step-dir>/videos/playback.mp4`.

---

## Key v4 metrics (at 20 M env-steps)

| metric | value |
|---|---|
| `Curriculum/terrain_levels` | peaked 5.5, settled ~4 |
| `env/lin_vel_tracking_error` | 0.40 (down from 0.65–0.7 wall in v1–v3) |
| `env/ang_vel_tracking_error` | 0.30 |
| `env/base_height` | 0.32–0.35 |
| `env/orientation_error` | 0.18 |
| `temperature/value` (α) | settled ~0.005–0.01 |
| `actor/entropy` | ~ −17.7 (target −16, σ=0.12) |
