# v4 inference checkpoint

20 M env-step weights from the `flashsac_full_v4` WandB run (2026-05-06).

## Contents

| file | size | purpose |
|---|---|---|
| `config.json` | 1.1 KB | full `FlashSACConfig` used at training time |
| `env_overrides.json` | 70 B | obs-restoration flags so play.py can reproduce the training-time obs space (lin_vel restored, no height_scan) |
| `command.txt` | 166 B | exact training CLI invocation |
| `step_19998720/actor.pt` | 3.3 MB | the policy network — sufficient for inference |
| `step_19998720/temperature.pt` | 3 KB | learned α |
| `step_19998720/reward_normalizer.pt` | 3.8 KB | reward stats (RMS) |
| `step_19998720/agent_state.pt` | 1.5 KB | minor agent metadata |

The critic files (`critic.pt` 26 MB, `target_critic.pt` 8.5 MB) are **not** included — they're only needed for resuming training, not for playback.

## Usage

```bash
python launchers/go2w_flashsac/play.py \
  --checkpoint launchers/go2w_flashsac/v4_checkpoint/step_19998720 \
  --num_envs 1 --num_episodes 5 --device cuda:0 --headless
```

```bash
python launchers/go2w_flashsac/create_video.py \
  launchers/go2w_flashsac/v4_checkpoint/step_19998720 \
  --num_envs 3 --duration 60
```

## Reproduction

See `launchers/go2w_flashsac/V4_REPRODUCE.md` for the full from-scratch recipe (clone repos, apply patches, install Isaac Lab, train).
