# DRL Quad Locomotion — Go2W

Deep reinforcement learning for the Unitree Go2W wheeled quadruped using [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [SBX (JAX)](https://github.com/araffin/sbx).

## Structure

```
launchers/
  go2w_PPO_SBX/       PPO training & evaluation (JAX backend via SBX)
  go2w_SAC_sbx/       SAC training & evaluation (JAX backend via SBX)

task_configs/         Modified robot_lab task config files (drop-in replacements)
  source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/
    rough_env_cfg.py              Active reward config (PPO-tested)
    rough_env_cfg_ppo_tested.py   Saved snapshot — PPO achieved stable locomotion
    flat_env_cfg.py               Flat terrain variant
    __init__.py                   Gym environment registrations
```

## Dependencies

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — base task framework
- [SBX](https://github.com/araffin/sbx) — Stable Baselines JAX
- Isaac Sim 4.x

## Usage

```bash
# PPO training
cd ~/robotics/IsaacLab
./isaaclab.sh -p /path/to/launchers/go2w_PPO_SBX/train.py --num_envs 2048 --headless --timesteps 50000000

# SAC training
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/train.py --num_envs 2048 --headless --timesteps 200000000

# Evaluation (auto-selects latest checkpoint)
./isaaclab.sh -p /path/to/launchers/go2w_PPO_SBX/play.py --num_envs 16
./isaaclab.sh -p /path/to/launchers/go2w_SAC_sbx/play.py --num_envs 16
```

## Status

- **v0** — PPO achieves stable locomotion with velocity tracking. SAC tuning in progress.
