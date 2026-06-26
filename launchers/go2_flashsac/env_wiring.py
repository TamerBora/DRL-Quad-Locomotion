"""Shared, metadata-derived obs/action wiring + self-check for the Go2 FlashSAC
wrapper (used by train.py and play.py so the two copies can't drift).

The wrapper builds the FlashSAC "full obs" = [proprio_history(225),
base_lin_vel, height_scan] by pulling base_lin_vel and height_scan OUT of the
IsaacLab `critic` observation group. Historically those columns were sliced
with HARDCODED offsets (base_lin_vel = critic[:, 0:3], height_scan =
critic[:, 48:]) and the per-joint action scale was hardcoded (0.125 hip /
0.25 other). Those assumptions are correct for the robot_lab AND the current
Fable critic — but if anyone adds a privileged critic term (actuator gains,
friction, mass, delay mask, ...) the hardcoded slice silently grabs the wrong
columns and the critic trains on garbage with NO crash.

This module derives every offset BY NAME from the observation manager's
per-term metadata (`active_terms` + `group_obs_term_dim`, confirmed present in
the installed IsaacLab) and asserts loudly on any mismatch.
"""

from __future__ import annotations

import numpy as np

SINGLE_FRAME_PROPRIO = 45  # ang_vel(3)+grav(3)+cmd(3)+jpos(12)+jvel(12)+act(12)


def resolve_group_layout(obs_mgr, group: str) -> tuple[dict[str, tuple[int, int]], int]:
    """{term_name: (start_index, dim)} and the group total dim, from metadata."""
    if group not in obs_mgr.active_terms:
        raise RuntimeError(f"obs group '{group}' not found; have {list(obs_mgr.active_terms)}")
    names = obs_mgr.active_terms[group]
    dims = [int(np.prod(d)) for d in obs_mgr.group_obs_term_dim[group]]
    layout: dict[str, tuple[int, int]] = {}
    start = 0
    for n, d in zip(names, dims):
        layout[n] = (start, d)
        start += d
    return layout, start


def resolve_action_scale(env, joint_order: list[str]) -> list[float]:
    """Per-joint action scale read from the live JointPositionAction term."""
    term = env.action_manager.get_term("joint_pos")
    scale = getattr(term, "_scale", None)
    if scale is None:
        raise RuntimeError("joint_pos action term has no _scale; cannot verify action scale")
    if hasattr(scale, "shape"):
        s = (scale[0] if scale.dim() > 1 else scale).detach().cpu().numpy().astype(float).tolist()
    else:
        s = [float(scale)] * len(joint_order)
    if len(s) != len(joint_order):
        raise RuntimeError(f"action scale dim {len(s)} != {len(joint_order)} joints")
    return s


def self_check(env, actor_obs_dim: int, obs_history: int, joint_order: list[str],
               verbose: bool = True) -> dict:
    """Derive + ASSERT the full obs/action wiring. Aborts loudly on any mismatch.

    Returns the derived wiring the wrapper needs:
      blv = (start, dim) of base_lin_vel in the critic group
      height_scan = (start, dim) or None (flat task)
      critic_total, action_scale, full_obs_dim
    """
    om = env.observation_manager

    # ── actor (policy) must be blind: 45*H, no privileged terms ──────────
    pol_layout, pol_total = resolve_group_layout(om, "policy")
    single = actor_obs_dim // max(1, obs_history)
    if pol_total != actor_obs_dim:
        raise RuntimeError(f"policy(actor) dim {pol_total} != reported {actor_obs_dim}")
    if single != SINGLE_FRAME_PROPRIO:
        raise RuntimeError(f"single-frame proprio {single} != {SINGLE_FRAME_PROPRIO} "
                           f"(actor {actor_obs_dim} / history {obs_history})")
    for forbidden in ("base_lin_vel", "height_scan"):
        if forbidden in pol_layout:
            raise RuntimeError(f"BLIND-ACTOR VIOLATION: '{forbidden}' is in the policy(actor) "
                               "group — actor must be proprioception-only.")

    # ── critic: locate base_lin_vel + height_scan BY NAME ────────────────
    crit_layout, crit_total = resolve_group_layout(om, "critic")
    if "base_lin_vel" not in crit_layout:
        raise RuntimeError(f"critic group has no 'base_lin_vel'; terms={list(crit_layout)}")
    bs, bd = crit_layout["base_lin_vel"]
    if bd != 3:
        raise RuntimeError(f"critic base_lin_vel dim {bd} != 3")
    hs = None
    if "height_scan" in crit_layout:
        hstart, hdim = crit_layout["height_scan"]
        if hdim <= 0:
            raise RuntimeError(f"critic height_scan dim {hdim} <= 0")
        # cross-check against the scanner's actual ray count if available
        try:
            scanner = env.scene["height_scanner"]
            n_rays = int(scanner.num_rays) if hasattr(scanner, "num_rays") else hdim
            if n_rays != hdim:
                raise RuntimeError(f"height_scan obs dim {hdim} != scanner rays {n_rays}")
        except KeyError:
            pass
        hs = (hstart, hdim)

    scale = resolve_action_scale(env, joint_order)
    full_obs_dim = actor_obs_dim + bd + (hs[1] if hs else 0)

    if verbose:
        print("=" * 72)
        print("[env_wiring] STARTUP SELF-CHECK (metadata-derived; aborts on mismatch)")
        print(f"  actor(policy) dim      : {pol_total}   (history {obs_history} x {single})")
        print(f"  actor terms            : {list(pol_layout)}")
        print(f"  critic total dim       : {crit_total}")
        print(f"  critic base_lin_vel    : start={bs} dim={bd}")
        print(f"  critic height_scan     : {('start=%d dim=%d' % hs) if hs else 'ABSENT (flat)'}")
        print(f"  critic terms           : {list(crit_layout)}")
        print(f"  action per-joint scale : {scale}")
        print(f"  FlashSAC full obs dim  : {full_obs_dim}  (= {actor_obs_dim} + {bd} + "
              f"{hs[1] if hs else 0})")
        print("=" * 72)

    return {
        "blv": (bs, bd),
        "height_scan": hs,
        "critic_total": crit_total,
        "action_scale": scale,
        "full_obs_dim": full_obs_dim,
        "policy_layout": pol_layout,
        "critic_layout": crit_layout,
    }
