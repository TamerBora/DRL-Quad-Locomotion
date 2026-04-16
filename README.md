# Go2W Locomotion — SAC, GSAC, PPO with IsaacLab

Deep reinforcement learning for the Unitree Go2W wheeled quadruped using [IsaacLab](https://github.com/isaac-sim/IsaacLab), [robot_lab](https://github.com/fan-ziqi/robot_lab), and [SBX (JAX)](https://github.com/araffin/sbx).

---

## Tested versions

| Dependency | Version |
|---|---|
| IsaacLab | v2.2.1 (commit `47780cf`) |
| robot_lab | v2.2.1 (commit `881036f`) |
| sbx-rl | 0.25.0 |
| stable-baselines3 | 2.8.0a4 |
| wandb | 0.25.1 |

---

## Setup

**1. Install IsaacLab**

Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

**2. Install robot_lab**

```bash
git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab
# activate your IsaacLab Python environment, then:
pip install -e source/robot_lab
```

**3. Clone this repo and install dependencies**

```bash
git clone <this-repo-url>
cd <repo-dir>
pip install -r requirements.txt
```

**4. Run setup script**

Copies task configs into robot_lab and installs the GSAC patch into sbx. Run from inside your IsaacLab Python environment.

```bash
# If robot_lab is at ~/robotics/robot_lab (default):
./setup_task_configs.sh

# If robot_lab is elsewhere:
ROBOT_LAB_DIR=/path/to/robot_lab ./setup_task_configs.sh
```

**5. Set ROBOT_LAB_DIR (if not at default location)**

```bash
export ROBOT_LAB_DIR=/path/to/robot_lab
```

---

## Training

All commands are run from the **IsaacLab root directory**.

### SAC — flat terrain

```bash
cd /path/to/IsaacLab

./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --algorithm sac --ml_framework jax \
  --num_envs 4096 --headless \
  --wandb_name my_run
```

### SAC — rough terrain

```bash
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --algorithm sac --ml_framework jax \
  --num_envs 4096 --headless \
  --wandb_name my_run
```

### PPO — flat terrain

```bash
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --algorithm ppo --ml_framework jax \
  --num_envs 4096 --headless \
  --wandb_name my_run
```

### PPO — rough terrain

```bash
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --algorithm ppo --ml_framework jax \
  --num_envs 4096 --headless \
  --wandb_name my_run
```

### Guided SAC (GSAC) — flat terrain

GSAC uses a privileged guide actor alongside the control actor. It requires the sbx GSAC patch installed by `setup_task_configs.sh`.

```bash
./isaaclab.sh -p /path/to/launchers/go2w_GSAC_sbx/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --num_envs 4096 --headless \
  --guidance_weight 0.5 \
  --wandb_name my_run
```

Disable WandB:

```bash
WANDB_MODE=disabled ./isaaclab.sh -p ...
```

Checkpoints are saved to `logs/sb3/<task>/<run-timestamp>/` (SAC/PPO) or `logs/gsac/<task>/<run-timestamp>/` (GSAC).

---

## Play

```bash
# SAC/PPO — auto-finds latest checkpoint
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --algorithm sac --ml_framework jax --num_envs 16

# SAC/PPO — explicit checkpoint
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --algorithm sac --ml_framework jax --num_envs 16 \
  --checkpoint /path/to/logs/.../model.zip

# GSAC
./isaaclab.sh -p /path/to/launchers/go2w_GSAC_sbx/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 \
  --num_envs 16 \
  --checkpoint /path/to/logs/.../model.zip

# Override velocity command (lin_x lin_y ang_z)
... --cmd 1.0 0.0 0.0
```

---

## Evaluation

Evaluates a checkpoint across 4 terrain types simultaneously (flat, uneven, stairs up, stairs down).

```bash
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/evaluate.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --algorithm sac --ml_framework jax \
  --checkpoint /path/to/logs/.../model.zip \
  --num_envs 32 --eval_steps 3000
```

---

## Repository structure

```
launchers/
  go2w_SAC_sbx/
    train.py          SAC and PPO training (select with --algorithm)
    play.py           SAC and PPO play
    evaluate.py       Multi-terrain evaluation

  go2w_GSAC_sbx/
    train.py          Guided SAC training
    play.py           Guided SAC play

  sbx_source/gsac/    GSAC implementation — patched into sbx by setup_task_configs.sh

task_configs/
  unitree_go2w/
    __init__.py       Gym task registrations
    flat_env_cfg.py   Flat terrain environment config
    rough_env_cfg.py  Rough terrain environment config
    agents/
      sb3_sac_cfg.yaml
      sb3_ppo_cfg.yaml
  unitree_a1/
    flat_env_cfg.py
    rough_env_cfg.py

setup_task_configs.sh   One-step setup: installs configs + GSAC patch
requirements.txt        Pinned Python dependencies
```
