# Go2W Locomotion — PPO & SAC with IsaacLab

Deep reinforcement learning for the **Unitree Go2W** wheeled quadruped using [IsaacLab](https://github.com/isaac-sim/IsaacLab), [robot_lab](https://github.com/fan-ziqi/robot_lab), and [SBX (JAX)](https://github.com/araffin/sbx).

Both PPO and SAC achieve flat-terrain forward locomotion. SAC learns faster but shows occasional instability; PPO is slower but more reliable.

---

## Requirements

Install these before using this repo:

| Dependency | Notes |
|-----------|-------|
| [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) | Follow official installation guide |
| [robot_lab](https://github.com/fan-ziqi/robot_lab) | Clone and install via `pip install -e .` inside the IsaacLab Python env |
| [SBX](https://github.com/araffin/sbx) | `pip install sbx-rl` inside the IsaacLab Python env |
| [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | `pip install stable-baselines3` |
| [WandB](https://wandb.ai) | `pip install wandb` — used for logging (set `WANDB_MODE=disabled` to skip) |

---

## Setup

```bash
# 1. Clone this repo
git clone <this-repo-url>
cd <repo-dir>

# 2. Copy modified task configs into your robot_lab installation
#    (defaults to ~/robotics/robot_lab — override with ROBOT_LAB_DIR if needed)
ROBOT_LAB_DIR=/path/to/robot_lab ./setup_task_configs.sh

# 3. (Optional) If robot_lab is not at ~/robotics/robot_lab, export the variable
#    so the launchers can find it at runtime too:
export ROBOT_LAB_DIR=/path/to/robot_lab
```

The `setup_task_configs.sh` script copies the reward configs and agent YAML files from `task_configs/` into the correct locations inside your `robot_lab` installation. Run it again after any config change.

---

## Training

All commands are run from inside the **IsaacLab root directory** using `./isaaclab.sh -p`.

### SAC (recommended — faster convergence)

```bash
cd ~/robotics/IsaacLab

./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/train.py \
    --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
    --algorithm sac \
    --ml_framework jax \
    --num_envs 2048 \
    --headless
```

Expected training time: ~25M timesteps to stable locomotion (~2–3 hours on a modern GPU).

### PPO

```bash
./isaaclab.sh -p /path/to/launchers/go2w_PPO_SBX/train.py \
    --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
    --algorithm ppo \
    --ml_framework jax \
    --num_envs 2048 \
    --headless
```

Expected training time: ~50M timesteps (~3–4 hours on a modern GPU).

### WandB logging

Training logs to WandB by default. To disable:

```bash
WANDB_MODE=disabled ./isaaclab.sh -p launchers/go2w_SAC_sbx/train.py ...
```

Checkpoints are saved to `logs/sb3/<task>/<run-timestamp>/`.

---

## Evaluation (play)

```bash
# SAC — auto-finds best checkpoint in logs/
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/play.py \
    --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
    --algorithm sac \
    --ml_framework jax \
    --num_envs 16

# SAC — specify a checkpoint explicitly
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/play.py \
    --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
    --algorithm sac \
    --ml_framework jax \
    --num_envs 16 \
    --checkpoint /path/to/logs/.../model.zip

# Override velocity command (lin_x lin_y ang_z in m/s, rad/s)
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/play.py \
    --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
    --algorithm sac \
    --ml_framework jax \
    --num_envs 16 \
    --cmd 1.0 0.0 0.0
```

---

## Repository Structure

```
launchers/
  go2w_SAC_sbx/
    train.py          SAC training script (JAX backend via SBX)
    play.py           SAC evaluation script

  go2w_PPO_SBX/
    train.py          PPO training script (JAX backend via SBX)
    play.py           PPO evaluation script

task_configs/         Drop-in replacements for robot_lab task configs
  unitree_go2w/
    __init__.py       Gym task registrations (RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0, ...)
    flat_env_cfg.py   Flat terrain env config (used for training)
    rough_env_cfg.py  Rough terrain env config
    agents/
      sb3_sac_cfg.yaml  SAC hyperparameters (learning_starts=10000, gradient_steps=512, ent_coef=auto)
      sb3_ppo_cfg.yaml  PPO hyperparameters

  unitree_a1/
    flat_env_cfg.py   A1 flat terrain config (used for baseline comparison)
    rough_env_cfg.py  A1 rough terrain config

setup_task_configs.sh   Installs task_configs into robot_lab (run once after clone)
```

---

## Key Design Decisions

### Why SAC works for Go2W

Previous SAC runs failed due to three compounding issues:

1. **`base_lin_vel = None`** — robot was blind to its own velocity, making velocity tracking structurally impossible
2. **`learning_starts = 100`** — gradient updates started on a nearly empty replay buffer, leading to degenerate Q-value estimates
3. **Reward imbalance** — the `upward` stability reward dominated tracking, so standing still was more profitable than moving

The current config fixes all three:
- `base_lin_vel` is included in observations
- `learning_starts = 10000` ensures a full replay buffer before the first gradient update
- `upward` weight is tuned so forward locomotion earns more than standing still

### SAC hyperparameters (key values)

```yaml
gradient_steps: 512    # UTD ratio = 512 — high sample efficiency
learning_starts: 10000 # wait for buffer to fill before first update
ent_coef: "auto_0.002" # automatic entropy tuning, initialized at 0.002
net_arch: [128, 128, 128]
```

### VecNormalize

Both scripts apply `VecNormalize` when `normalize_input: true` is set in the agent YAML. The normalization statistics are saved alongside the model checkpoint (`model_vecnormalize.pkl`) and restored automatically during play.

