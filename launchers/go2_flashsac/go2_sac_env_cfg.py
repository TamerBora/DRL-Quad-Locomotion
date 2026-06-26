"""Separate SAC task copy for the legged Go2 — faithful RSL-RL-SAC reproduction.

Subclasses the original `UnitreeGo2RoughEnvCfg` and registers a new gym id
`RobotLab-Isaac-Velocity-Rough-Unitree-Go2-SAC-v0`. The original PPO task is
left untouched.

Per the ETH RSL-RL-SAC paper (arXiv:2605.24975), SAC matches PPO on legged
locomotion with **no reward changes** — the fixes are on the SAC side (action
bounds, init, timeout bootstrap) plus the right hyperparameters. So this cfg
keeps robot_lab's Go2 reward/terminations **verbatim** and only:

  * gives the actor a FULLY BLIND, proprioception-only observation — NO
    `base_lin_vel` (no velocity-estimator dependence) and NO `height_scan`. To
    recover velocity observability it stacks a 5-frame **history** of the
    proprioceptive obs (the actor infers velocity from proprioceptive dynamics).
    `base_lin_vel` + `height_scan` stay in the privileged critic only →
    asymmetric actor/critic. Deployable from IMU + joint encoders alone.
  * starts the curriculum at terrain level 0 and spawns upright (yaw-only),
    which are training-stability choices, not reward engineering.

(`bad_orientation` / reward shaping are fallback levers handled in train.py if
the faithful reproduction underperforms — not applied here.)
"""

import copy

import gymnasium as gym
from isaaclab.utils import configclass

# Importing the rough_env_cfg submodule also runs unitree_go2/__init__.py, which
# registers the original Go2 ids (used by evaluate.py for the PPO baseline).
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

SAC_TASK_ID = "RobotLab-Isaac-Velocity-Rough-Unitree-Go2-SAC-v0"
OBS_HISTORY = 5  # proprioceptive frames stacked for the blind actor


@configclass
class UnitreeGo2RoughSACEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # Applies all the stock Go2 reward weights, nulls policy base_lin_vel /
        # height_scan, sets illegal_contact=None, etc. (robot_lab defaults).
        super().__post_init__()

        # ── Fully blind actor: proprioception only, 5-frame history ──────
        # robot_lab already nulls base_lin_vel + height_scan from the policy
        # group → 45-dim proprio (base_ang_vel, projected_gravity, commands,
        # joint_pos, joint_vel, last_action). We keep them None (no velocity
        # estimator, no terrain scan) and stack OBS_HISTORY frames so the actor
        # can infer base velocity from proprioceptive dynamics. base_lin_vel +
        # height_scan remain in the privileged critic group only.
        self.observations.policy.history_length = OBS_HISTORY
        self.observations.policy.flatten_history_dim = True

        # ── Training-stability (non-reward) choices ──────────────────────
        # Spawn all envs at the easiest terrain; per-env curriculum promotes.
        self.scene.terrain.max_init_terrain_level = 0
        # Upright (yaw-only) spawn, no roll/pitch spin (user requirement; also
        # avoids wasting episodes on upside-down spawns).
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.5, 0.5),
        }
        # Per-episode actuator-gain randomisation makes a multimodal Q-landscape
        # SAC's single critic struggles with; disable during training.
        self.events.randomize_actuator_gains = None

        # ── Ease the curriculum: remove stairs (blind actor can't anticipate) ──
        # The default rough terrain is ~40% stairs (pyramid_stairs +
        # pyramid_stairs_inv). A blind/proprioceptive policy can't see steps
        # coming, which likely caps the terrain level. Drop the two stair
        # sub-terrains, leaving boxes + random_rough + slopes (all reactively
        # traversable). Deep-copy first so the shared ROUGH_TERRAINS_CFG (used by
        # the original PPO task and others) is NOT mutated.
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg = copy.deepcopy(tg)
            for _k in ("pyramid_stairs", "pyramid_stairs_inv"):
                tg.sub_terrains.pop(_k, None)
            self.scene.terrain.terrain_generator = tg

        # NB: reward weights and terminations are left exactly as robot_lab's
        # Go2 PPO task defines them (faithful "reuse PPO's reward").

        # Prune zero-weight reward terms. The parent only does this when the
        # class name matches exactly "UnitreeGo2RoughEnvCfg", so our subclass
        # must call it — otherwise inactive wheeled-robot terms (e.g.
        # wheel_vel_penalty, whose joints don't exist on the legged Go2) remain
        # and crash observation/reward parsing.
        self.disable_zero_weight_rewards()


gym.register(
    id=SAC_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeGo2RoughSACEnvCfg,
        "rsl_rl_cfg_entry_point": (
            "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped."
            "unitree_go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
        ),
    },
)
