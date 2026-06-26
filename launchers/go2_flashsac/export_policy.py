"""Export a trained FlashSAC Go2 actor to TorchScript + ONNX for hardware deploy.

The exported module is the deterministic policy:

    obs(45) ──FlashSACActor.get_mean_and_std──▶ mean ──tanh──▶ ×ACTION_BOUNDS ──▶ action(12)

i.e. exactly what the training/play wrapper feeds into IsaacLab's action manager.
On the robot, the joint targets are then:

    q_target = q_default + joint_scale * action

(see DEPLOY.md for the full obs/action contract). This script does NOT launch
Isaac Sim — it loads the actor weights directly and traces them.

Usage:
    python export_policy.py --checkpoint logs/flashsac/<task>/<run>/best
    python export_policy.py --checkpoint <dir> --output <dir> --device cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "FlashSAC"))

from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402

# ── Deployment constants (Go2 rough_env_cfg) ────────────────────────────────
# Joint order as defined in unitree_go2/rough_env_cfg.py (the policy obs and
# action vectors follow this order).
JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# Per-joint action scale: hip 0.125, thigh/calf 0.25.
JOINT_ACTION_SCALE = [
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
]
# Default joint angles (UNITREE_GO2_CFG.init_state.joint_pos): hip 0, thigh 0.8, calf -1.5.
DEFAULT_JOINT_POS = [
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
]
# Single proprioceptive frame (45-dim), in order, with scales. The actor input
# is OBS_HISTORY of these frames flattened per-term (no base_lin_vel, no height_scan).
OBS_TERMS = [
    {"name": "base_ang_vel", "dim": 3, "scale": 0.25},
    {"name": "projected_gravity", "dim": 3, "scale": 1.0},
    {"name": "velocity_commands", "dim": 3, "scale": 1.0},
    {"name": "joint_pos_rel", "dim": 12, "scale": 1.0, "note": "joint_pos - default_joint_pos, JOINT_ORDER"},
    {"name": "joint_vel_rel", "dim": 12, "scale": 0.05, "note": "JOINT_ORDER"},
    {"name": "last_action", "dim": 12, "scale": 1.0, "note": "previous raw policy action a=b+c*tanh, JOINT_ORDER"},
]
CONTROL_HZ = 50.0   # decimation 4 × sim.dt 0.005 = 0.02 s
SIM_DT = 0.005
DECIMATION = 4


class _RMSNormDeploy(nn.Module):
    """Export-friendly RMSNorm: F.rms_norm decomposed into primitive ops so
    ONNX (which lacks aten::rms_norm before opset 23) and any tracer can
    handle it. Numerically identical to flash_rl's UnitRMSNorm."""

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def _decompose_rms_norm(module: nn.Module) -> None:
    """Replace UnitRMSNorm children with the export-friendly decomposition."""
    from flash_rl.agents.flashSAC.layer import UnitRMSNorm
    for name, child in module.named_children():
        if isinstance(child, UnitRMSNorm):
            setattr(module, name, _RMSNormDeploy(child.weight.data, child.eps))
        else:
            _decompose_rms_norm(child)


class DeployActor(nn.Module):
    """Deterministic policy: obs(48) → action(12) = b + c·tanh(mean) (per-joint),
    where b=(a_max+a_min)/2, c=(a_max-a_min)/2 are the per-joint action bounds."""

    def __init__(self, actor: FlashSACActor, a_offset: torch.Tensor, a_scale: torch.Tensor) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("a_offset", a_offset)
        self.register_buffer("a_scale", a_scale)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor.get_mean_and_std(observations, False)
        return self.a_offset + self.a_scale * torch.tanh(mean)


def _strip_compile_prefix(state: dict) -> dict:
    """Checkpoints are saved from a torch.compile'd module (keys prefixed
    `_orig_mod.`). Strip it so the bare FlashSACActor loads strict=True."""
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }


def _load_actor(ckpt_dir: Path, device: torch.device) -> tuple[FlashSACActor, int, int]:
    # `best/`, `final/` and `step_*` all sit one level under the run dir,
    # where config.json lives.
    cfg_json = ckpt_dir.parent / "config.json"
    if not cfg_json.exists():
        raise FileNotFoundError(f"config.json not found next to checkpoint dir {ckpt_dir}")
    cfg = json.loads(cfg_json.read_text())

    actor_ckpt = torch.load(ckpt_dir / "actor.pt", map_location="cpu", weights_only=False)
    state = _strip_compile_prefix(actor_ckpt.get("network_state_dict", actor_ckpt))

    # Infer dims directly from the weights (robust to task drift).
    obs_dim = None
    act_dim = None
    for k, v in state.items():
        if k.endswith("embedder.norm.running_mean"):
            obs_dim = int(v.shape[0])
        if k.endswith("predictor.mean_head.bias"):
            act_dim = int(v.shape[0])
    if obs_dim is None or act_dim is None:
        raise RuntimeError(f"Could not infer obs/action dims from actor.pt (obs={obs_dim}, act={act_dim})")

    actor = FlashSACActor(
        num_blocks=int(cfg["actor_num_blocks"]),
        input_dim=obs_dim,
        hidden_dim=int(cfg["actor_hidden_dim"]),
        action_dim=act_dim,
    ).to(device)
    actor.load_state_dict(state)
    actor.eval()
    return actor, obs_dim, act_dim


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a FlashSAC Go2 actor for deployment.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint directory containing actor.pt (e.g. .../best).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir. Default: <checkpoint>/exported/")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for export/validation (cpu is fine and most portable).")
    parser.add_argument("--no_onnx", action="store_true", help="Skip ONNX export.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).expanduser().resolve()
    if not (ckpt_dir / "actor.pt").exists():
        sys.exit(f"Missing actor.pt under: {ckpt_dir}")
    out_dir = Path(args.output) if args.output else (ckpt_dir / "exported")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Per-joint action bounds + obs history this checkpoint was trained with (sidecar).
    ov = ckpt_dir.parent / "env_overrides.json"
    a_min = a_max = None
    obs_history = 1
    if ov.exists():
        _ov = json.loads(ov.read_text())
        ab = _ov.get("action_bounds", {})
        if isinstance(ab, dict):
            a_min, a_max = ab.get("a_min"), ab.get("a_max")
        obs_history = int(_ov.get("obs_history", 0) or 1)
    if a_min is None or a_max is None:
        sys.exit("env_overrides.json missing per-joint action_bounds (a_min/a_max).")
    a_min_t = torch.tensor(a_min, dtype=torch.float32, device=device)
    a_max_t = torch.tensor(a_max, dtype=torch.float32, device=device)
    a_offset = 0.5 * (a_max_t + a_min_t)
    a_scale = 0.5 * (a_max_t - a_min_t)

    actor, obs_dim, act_dim = _load_actor(ckpt_dir, device)
    _decompose_rms_norm(actor)  # export-friendly RMSNorm (ONNX + clean trace)
    deploy = DeployActor(actor, a_offset, a_scale).to(device).eval()
    single_frame = obs_dim // max(1, obs_history)
    print(f"[export] loaded actor: obs_dim={obs_dim} (proprio history={obs_history}, "
          f"single_frame={single_frame})  action_dim={act_dim}  (fully blind)")
    if single_frame != 45:
        print(f"[export] note: single_frame={single_frame} (expected 45 for the blind proprio "
              f"actor: ang_vel+grav+cmd+jpos+jvel+last_action).")

    dummy = torch.zeros(1, obs_dim, device=device)
    with torch.inference_mode():
        ref = deploy(dummy).cpu()

    # ── TorchScript (trace) ─────────────────────────────────────────────
    ts_path = out_dir / "actor.ts"
    with torch.inference_mode():
        traced = torch.jit.trace(deploy, dummy, check_trace=False)
    traced.save(str(ts_path))
    with torch.inference_mode():
        ts_out = torch.jit.load(str(ts_path)).to(device)(dummy).cpu()
    ts_err = (ts_out - ref).abs().max().item()
    print(f"[export] TorchScript → {ts_path}   max|Δ| vs live = {ts_err:.2e}")

    # ── ONNX ────────────────────────────────────────────────────────────
    onnx_err = None
    if not args.no_onnx:
        onnx_path = out_dir / "actor.onnx"
        try:
            torch.onnx.export(
                deploy, dummy, str(onnx_path),
                input_names=["obs"], output_names=["action"],
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                opset_version=17,
            )
            try:
                import onnxruntime as ort  # type: ignore
                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                onnx_out = torch.as_tensor(
                    sess.run(["action"], {"obs": dummy.cpu().numpy()})[0]
                )
                onnx_err = (onnx_out - ref).abs().max().item()
                print(f"[export] ONNX        → {onnx_path}   max|Δ| vs live = {onnx_err:.2e}")
            except ImportError:
                print(f"[export] ONNX        → {onnx_path}   (onnxruntime not installed; skipped numeric check)")
        except Exception as e:  # noqa: BLE001
            print(f"[export] ONNX export failed ({e}); TorchScript is still available.")

    # ── Deployment spec sidecar ─────────────────────────────────────────
    spec = {
        "checkpoint": str(ckpt_dir),
        "obs_dim": obs_dim,
        "obs_history": obs_history,
        "obs_single_frame_dim": single_frame,
        "obs_note": (f"Fully-blind actor: {obs_history} stacked frames of the 45-dim proprio obs "
                     "[base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos_rel(12), "
                     "joint_vel_rel(12), last_action(12)]; NO base_lin_vel, NO height_scan. "
                     "Flatten per-term oldest->newest (IsaacLab ObservationGroup history); keep a "
                     "per-term ring buffer of the last N frames on the robot."),
        "action_dim": act_dim,
        "action_a_min": a_min,
        "action_a_max": a_max,
        "control_hz": CONTROL_HZ,
        "sim_dt": SIM_DT,
        "decimation": DECIMATION,
        "joint_order": JOINT_ORDER,
        "joint_action_scale": JOINT_ACTION_SCALE,
        "default_joint_pos": DEFAULT_JOINT_POS,
        "obs_terms": OBS_TERMS,
        "action_pipeline": ("the exported module already outputs action[j] = b_j + c_j*tanh(mean_j) "
                            "with b=(a_max+a_min)/2, c=(a_max-a_min)/2; then "
                            "q_target[j] = default_joint_pos[j] + joint_action_scale[j] * action[j]."),
        "validation_max_abs_err": {"torchscript": ts_err, "onnx": onnx_err},
    }
    spec_path = out_dir / "deploy_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"[export] spec        → {spec_path}")
    print(f"[export] done. Artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
