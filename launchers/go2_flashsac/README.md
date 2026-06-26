# go2_flashsac

FlashSAC training, evaluation, and deployment-export pipeline for the **regular Unitree Go2** (legged quadruped) on rough-terrain locomotion, built on [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [robot_lab](https://github.com/fan-ziqi/robot_lab).

This is the **deployable-actor** sibling of `go2w_flashsac`. The actor is strictly proprioceptive (45-dim, no body-velocity sensor, no height scan) so the trained policy can run directly on the physical Go2; the critic is privileged (sees `base_lin_vel` + `height_scan`) — the same asymmetry the PPO baseline used to reach mean terrain level ~4.

## Task

`RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` — velocity tracking on procedurally generated terrain (pyramid stairs, inverted stairs, random boxes, rough, slopes). Terrain difficulty increases via curriculum as the robot improves. **Target: mean terrain level ≥ 4.**

## Observation layout (deployable asymmetric)

| Group | Dim | Contents |
|---|---|---|
| **actor** (deployable) | 45 | `base_ang_vel(3)` · `projected_gravity(3)` · `velocity_commands(3)` · `joint_pos(12)` · `joint_vel(12)` · `last_action(12)` |
| **critic** (privileged) | 48+H | actor obs + `base_lin_vel(3)` + `height_scan(H)` |

The env wrapper emits the full critic obs as `[policy(45), base_lin_vel(3), height_scan(H)]`; FlashSAC stores it in the buffer, feeds the full obs to the critic, and slices the first 45 dims for the actor (`asymmetric_observation=True`). The exported actor therefore consumes exactly the 45-dim deployable obs. See [DEPLOY.md](DEPLOY.md) for the full deployment contract.

## Requirements

- IsaacLab set up (`env_isaaclab` conda env with `robot_lab` installed)
- FlashSAC backbone present in `FlashSAC/` (vendored; patches in `flashsac_patches/` already applied)

```bash
conda activate env_isaaclab
```

## Training

```bash
cd /home/tamer/robotics/launchers/go2_flashsac
python train.py --wandb_name go2_flashsac_v1 --num_envs 256 --seed 42 --headless
```

| Argument | Default | Description |
|---|---|---|
| `--task` | `RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` | IsaacLab gym task ID |
| `--wandb_name` | `go2_flashsac_v1` | WandB run name |
| `--num_envs` | `256` | Parallel envs (RTX 4070 Laptop 8 GB) |
| `--total_steps` | `50_000_000` | Total environment steps (~4 h) |
| `--seed` | `42` | Random seed |
| `--headless` | — | Run without GUI |

Checkpoints land in `logs/flashsac/<task>/<run_name>_<timestamp>/` (every ~1M steps, plus `best/` and `final/`). `config.json` and `env_overrides.json` sidecars are saved for `play.py` / `evaluate.py` / `export_policy.py`.

WandB project: `flashsac_go2`. The success KPI is logged as **`env/mean_terrain_level`** (watch it climb to ≥4). Set `WANDB_MODE=offline` to disable online sync.

## Play / visualise

```bash
python play.py                                   # auto-discover latest checkpoint
python play.py --cmd 1.0 0.0 0.0                 # fixed forward command
python play.py --checkpoint logs/flashsac/.../best --real_time
```

## Evaluate vs PPO baseline

```bash
python evaluate.py                               # auto-finds latest PPO + FlashSAC ckpts
python evaluate.py --agent flashsac --num_episodes 200
```

Compares mean terrain level, tracking error, cost of transport, survival, etc. against the latest `unitree_go2_rough` PPO run. Results → `eval_results/compare_<ts>.json`.

## Export for hardware

```bash
python export_policy.py --checkpoint logs/flashsac/.../best
```

Emits TorchScript (`actor.ts`) and ONNX (`actor.onnx`) of the 45-dim→12-dim actor into `<checkpoint>/exported/`, validates them against the live actor, and writes the obs/action spec. See [DEPLOY.md](DEPLOY.md).

## Key design notes

**Deployable actor.** Unlike `go2w_flashsac` (which restored `base_lin_vel` into the *policy* obs to break a tracking plateau, making the actor non-deployable), this pipeline keeps the actor proprioceptive and routes the privileged info to the critic only. This is the structural fix for the prior peak-2 regression: the critic can finally score velocity tracking and terrain.

**Anti-regression overrides** (in `train.py`): spawn all envs at terrain level 0 (`max_init_terrain_level=0`), one velocity command per episode, bumped tracking weights, larger actor `(3,256)`, and `temp_target_sigma=0.12` to sustain exploration.

**Action bounds.** The actor outputs `tanh ∈ [-1,1]`, scaled by `5.0` before IsaacLab applies its per-joint scale (`0.125` hip / `0.25` other) and the default joint pose.

**Why a custom wrapper.** We launch Isaac Sim ourselves (needed for `robot_lab` task registration), so FlashSAC's built-in `IsaacLabVectorEnv` (which calls `AppLauncher` again) would crash. `Go2IsaacEnvWrapper` reproduces the interface without re-launching.
