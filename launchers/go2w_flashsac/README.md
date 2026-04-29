# go2w_flashsac

FlashSAC training and evaluation pipeline for the **Unitree Go2-W** (wheeled quadruped) on rough terrain locomotion, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [robot_lab](https://github.com/fan-ziqi/robot_lab).

## Task

`RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0` — velocity-tracking on procedurally generated terrain (pyramid stairs, inverted stairs, random boxes, rough, slopes). Terrain difficulty increases via curriculum as the robot improves.

## Requirements

- IsaacLab environment already set up (`env_isaaclab` conda env with `robot_lab` installed)
- Python 3.10 or 3.11
- The `go2w_SAC_sbx` pipeline from this repo working (confirms IsaacLab + robot_lab are wired up correctly)

## Setup

**1. Clone FlashSAC into this folder:**

```bash
cd go2w_flashsac
git clone https://github.com/Holiday-Robot/FlashSAC FlashSAC
```

**2. Activate the IsaacLab environment:**

```bash
conda activate env_isaaclab
```

No additional pip installs are needed — FlashSAC is imported directly via `sys.path` from the cloned directory.

## Training

```bash
python train.py --wandb_name flashsacv1 --num_envs 256 --seed 42 --headless
```

| Argument | Default | Description |
|---|---|---|
| `--task` | `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0` | IsaacLab gym task ID |
| `--wandb_name` | `flashsacv1` | WandB run name |
| `--num_envs` | `256` | Parallel environments (tuned for RTX 4070 Laptop 8 GB) |
| `--total_steps` | `50_000_000` | Total environment steps |
| `--seed` | `42` | Random seed |
| `--headless` | — | Run without GUI (recommended for training) |

Checkpoints are saved under `logs/flashsac/<task>/<run_name>_<timestamp>/` every ~1M steps and at the end of training. A `config.json` with the full agent architecture is also saved there — `play.py` reads it automatically.

WandB logs to project `flashsac_go2w`. Set `WANDB_MODE=offline` to disable online sync.

### Hardware notes (RTX 4070 Laptop, 8 GB VRAM)

The defaults are tuned for this GPU:

| Parameter | Value |
|---|---|
| `num_envs` | 256 |
| `buffer_max_length` | 500 000 |
| `sample_batch_size` | 1024 |
| `use_amp` | `True` |
| `use_compile` | `True` |

If you hit OOM, reduce `--num_envs 128` or open `train.py` and set `buffer_device_type="cpu"`.

## Evaluation / Play

```bash
# Auto-discover the latest checkpoint
python play.py

# Fix a forward velocity command
python play.py --cmd 1.0 0.0 0.0

# Specific checkpoint directory
python play.py --checkpoint logs/flashsac/.../step_5000000

# Throttle to real-time
python play.py --real_time
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | auto | Path to checkpoint dir (contains `actor.pt`). Auto-finds latest if omitted. |
| `--num_envs` | `16` | Parallel envs for visualisation |
| `--num_episodes` | `10` | Stop after this many completed episodes |
| `--cmd LIN_X LIN_Y ANG_Z` | env-sampled | Override velocity command every step |
| `--real_time` | off | Throttle stepping to sim dt |

## Key design notes

**Action bounds**: The actor outputs `tanh ∈ [-1, 1]`, which the wrapper scales by `5.0` before passing to IsaacLab — matching the SBX SAC baseline for this task (`Box(-5, 5)`). IsaacLab then applies its own internal joint scaling (×0.25 for leg joints, ×5.0 for wheel velocity).

**Why not `make_isaaclab_env()`**: FlashSAC's built-in `IsaacLabVectorEnv` calls `AppLauncher` internally. Since we launch Isaac Sim ourselves at the top of the script (required for robot_lab task registration), using their wrapper would try to start a second simulation instance and crash. The `Go2WIsaacEnvWrapper` class reproduces the same interface without re-launching.

**`config.json`**: Saved alongside checkpoints after the first run with the current `train.py`. Older checkpoints (without `config.json`) fall back to the hardcoded defaults in `play.py`.
