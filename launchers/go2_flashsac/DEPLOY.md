# Go2 FlashSAC — deployment contract

This document specifies exactly what the exported policy expects and produces, so it can run on the physical Unitree Go2. The exported actor is **proprioceptive and deployable** — it needs no body-velocity sensor and no height scan.

Produce the artifacts with:

```bash
python export_policy.py --checkpoint logs/flashsac/<task>/<run>/best
```

→ `<checkpoint>/exported/actor.ts` (TorchScript), `actor.onnx` (ONNX), `deploy_spec.json`.

## Control loop

- **Rate:** 50 Hz (`decimation=4` × `sim.dt=0.005` = 0.02 s per control step).
- Each step: build the 45-dim observation, run the policy, convert to joint targets, send to the PD controller.
- **PD gains** (from `UNITREE_GO2_CFG`): stiffness `Kp=25.0`, damping `Kd=0.5`, effort limit `23.5 N·m`.

## Joint order (12 DoF)

All joint-indexed vectors (obs `joint_pos`/`joint_vel`/`last_action`, and the action output) follow this order:

```
0 FR_hip   1 FR_thigh   2 FR_calf
3 FL_hip   4 FL_thigh   5 FL_calf
6 RR_hip   7 RR_thigh   8 RR_calf
9 RL_hip  10 RL_thigh  11 RL_calf
```

Default joint angles `q_default` (rad): hip `0.0`, thigh `0.8`, calf `-1.5` (repeated per leg).

## Observation — fully blind, proprioceptive history (45×H dims)

The actor is **fully blind/proprioceptive**: no `base_lin_vel` (no velocity-estimator dependence) and
no `height_scan`. It infers velocity from a stacked **history of `H` (=5)** proprioceptive frames.
`base_lin_vel` + `height_scan` are used **only by the critic during training** (privileged), never at
deploy. Deployable from IMU + joint encoders alone. Exact dims/scales/`H` are in `deploy_spec.json`.

**Single 45-dim proprio frame**, in order:

| # | term | dim | scale | notes |
|---|---|---|---|---|
| 0–2 | `base_ang_vel` | 3 | ×0.25 | base angular velocity (gyro) |
| 3–5 | `projected_gravity` | 3 | ×1.0 | gravity unit vector in body frame (IMU) |
| 6–8 | `velocity_commands` | 3 | ×1.0 | `[lin_x, lin_y, ang_z]` user command |
| 9–20 | `joint_pos_rel` | 12 | ×1.0 | `q - q_default`, JOINT_ORDER |
| 21–32 | `joint_vel_rel` | 12 | ×0.05 | joint velocity, JOINT_ORDER |
| 33–44 | `last_action` | 12 | ×1.0 | previous raw policy action (`a = b + c·tanh`), JOINT_ORDER |

The actor input is `H` frames flattened **per-term, oldest→newest** (IsaacLab `ObservationGroup`
history): `[ang_vel(t-4..t), grav(t-4..t), cmd(t-4..t), jpos(t-4..t), jvel(t-4..t), last_action(t-4..t)]`.

Notes:
- Apply each term's scale before concatenating (matches `velocity_env_cfg.py`).
- On the robot keep a per-term ring buffer of the last `H` frames; pad with the first frame at startup.
- `last_action` is the policy's own output, not the realized joint motion. Initialize to zeros.

## Action — 12 dims (per-joint asymmetric)

The exported module already applies `tanh` and the **per-joint** affine bounds, so its output is:

```
action[j] = b_j + c_j * clip(tanh(mean_j), -1, 1)
            b_j = (a_max_j + a_min_j)/2,  c_j = (a_max_j - a_min_j)/2
```

where `a_min/a_max` (per joint, in `deploy_spec.json`) come from the robot's soft joint limits
(`a = (q_soft_limit - q_default)/joint_action_scale`). Convert to joint **position targets**:

```
q_target[j] = q_default[j] + joint_action_scale[j] * action[j]
```

where `joint_action_scale` = `0.125` for hip joints and `0.25` for thigh/calf joints.

## Minimal runtime example (TorchScript, proprio history)

```python
import torch
from collections import deque
policy = torch.jit.load("exported/actor.ts").eval()  # outputs action = b + c·tanh already

H = 5                                              # obs_history (see deploy_spec.json)
q_default = torch.tensor([0,0.8,-1.5]*4)
scale     = torch.tensor([0.125,0.25,0.25]*4)
last_action = torch.zeros(12)
hist = {k: deque(maxlen=H) for k in ["av","g","cmd","jp","jv","la"]}

def control_step(gyro, proj_grav, cmd, q, qd):     # @ 50 Hz — NO base_lin_vel, NO height_scan
    global last_action
    frame = {"av": gyro*0.25, "g": proj_grav, "cmd": cmd,
             "jp": (q-q_default)*1.0, "jv": qd*0.05, "la": last_action}
    for k, v in frame.items():
        if not hist[k]:
            for _ in range(H): hist[k].append(v)   # startup: fill with first frame
        else:
            hist[k].append(v)
    obs = torch.cat([torch.cat(list(hist[k])) for k in ["av","g","cmd","jp","jv","la"]]).unsqueeze(0)  # (1, 225)
    with torch.inference_mode():
        action = policy(obs).squeeze(0)            # (12,) already b + c·tanh
    last_action = action
    return q_default + scale * action              # → PD position targets
```

All constants above are also written to `deploy_spec.json` by `export_policy.py`, and the exporter validates that the TorchScript/ONNX outputs match the live actor (printed `max|Δ|` should be ~1e-6).
