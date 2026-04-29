"""
FlashGSAC update functions.

Key differences from FlashSAC:
  - Control actor loss = SAC_loss(noisy_obs, guide_q) + λ · L1(a_c − stop_grad(a_g))
  - Guide actor loss   = standard SAC_loss(clean_obs)
  - Critic             = trained on clean (guide) observations; next-action bootstrap
                         uses the guide actor, not the control actor.
"""

from typing import Any, Optional

import torch
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.utils.network import Network
from flash_rl.buffers import Batch


def _prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


# --------------------------------------------------------------------------- #
# Shared compiled helpers (copied from flash_rl/agents/flashSAC/update.py)    #
# --------------------------------------------------------------------------- #

@torch.compile
def _select_min_q_log_probs(
    next_qs: torch.Tensor,       # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]  # (B, num_bins)
    return selected


@torch.compile
def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,            # (B,)
    done: torch.Tensor,              # (B,)
    actor_entropy: torch.Tensor,     # (B,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]
    reward        = reward.reshape(-1, 1)
    done          = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    bin_width  = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(min_v, max_v, num_bins,
                                device=target_log_probs.device,
                                dtype=target_log_probs.dtype).view(1, -1)

    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    b     = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    frac  = b - lower.float()

    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    target_probs = torch.zeros(batch_size, num_bins,
                               dtype=target_probs_exp.dtype,
                               device=target_probs_exp.device)
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)
    return target_probs


# --------------------------------------------------------------------------- #
# Control actor update — SAC loss + L1 guidance toward guide actor             #
# --------------------------------------------------------------------------- #

def update_actor(
    actor: Network,
    guide_actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    guidance_weight: float,
    bc_alpha: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """
    Control actor update.

    Loss = SAC(noisy_obs) + λ · L1(a_c − stop_grad(a_g))

    The critic evaluates Q(guide_obs, a_c) — the critic lives in the clean
    observation space, so we pass guide observations even though the actions
    come from the noisy-observation control actor.
    """
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        # Concatenate current + next actor obs so we sample actions for both in one pass.
        actor_obs_all = torch.cat(
            [batch["actor_observation"], batch["actor_next_observation"]], dim=0
        )
        actions_all, info_all = actor(observations=actor_obs_all, training=True)
        log_probs_all = info_all["log_prob"]

        # Current-step slice
        actions   = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        # Guide reference actions — no gradient flows into the guide actor.
        with torch.no_grad():
            guide_obs_all = torch.cat(
                [batch["guide_observation"], batch["guide_next_observation"]], dim=0
            )
            guide_actions_all, _ = guide_actor(observations=guide_obs_all, training=False)
            guide_actions = torch.chunk(guide_actions_all, 2, dim=0)[0]

        # Q evaluated at (guide_obs, control_action).
        # Critic lives in guide-obs space — always feed it guide observations.
        critic.network.requires_grad_(False)
        qs, _ = critic(
            observations=batch["guide_observation"],
            actions=actions,
            training=False,
        )
        q = torch.minimum(qs[0], qs[1])
        critic.network.requires_grad_(True)

        temp_value = temperature().detach()
        sac_loss   = (log_probs * temp_value - q).mean()

        # Optional BC conservatism term from FlashSAC (actor_bc_alpha > 0).
        if bc_alpha > 0:
            q_abs   = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            sac_loss = sac_loss + bc_alpha * q_abs * bc_loss

        # Guidance term: pull control actor toward guide actor's actions.
        guidance_loss = torch.abs(actions - guide_actions).mean()
        actor_loss    = sac_loss + guidance_weight * guidance_loss

        entropy     = -log_probs.mean()
        mean_action = actions.mean()

    assert actor.optimizer is not None
    actor.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(actor_loss).backward()
        grad_scaler.step(actor.optimizer)
        grad_scaler.update()
    else:
        actor_loss.backward()
        actor.optimizer.step()

    if actor.scheduler is not None:
        actor.scheduler.step()
    actor.normalize_parameters()

    return _prefix({
        "loss":          actor_loss,
        "sac_loss":      sac_loss,
        "guidance_loss": guidance_loss,
        "entropy":       entropy,
        "mean_action":   mean_action,
    }, "actor")


# --------------------------------------------------------------------------- #
# Guide actor update — standard SAC on clean observations                      #
# --------------------------------------------------------------------------- #

def update_guide_actor(
    guide_actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """
    Guide actor update — standard SAC loss using privileged (clean) observations.
    No guidance penalty; the guide is trained purely to maximise Q.
    """
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        guide_obs_all = torch.cat(
            [batch["guide_observation"], batch["guide_next_observation"]], dim=0
        )
        actions_all, info_all = guide_actor(observations=guide_obs_all, training=True)
        log_probs_all = info_all["log_prob"]

        actions   = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        critic.network.requires_grad_(False)
        qs, _ = critic(
            observations=batch["guide_observation"],
            actions=actions,
            training=False,
        )
        q = torch.minimum(qs[0], qs[1])
        critic.network.requires_grad_(True)

        temp_value       = temperature().detach()
        guide_actor_loss = (log_probs * temp_value - q).mean()
        entropy          = -log_probs.mean()

    assert guide_actor.optimizer is not None
    guide_actor.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(guide_actor_loss).backward()
        grad_scaler.step(guide_actor.optimizer)
        grad_scaler.update()
    else:
        guide_actor_loss.backward()
        guide_actor.optimizer.step()

    if guide_actor.scheduler is not None:
        guide_actor.scheduler.step()
    guide_actor.normalize_parameters()

    return _prefix({
        "loss":    guide_actor_loss,
        "entropy": entropy,
    }, "guide_actor")


# --------------------------------------------------------------------------- #
# Critic update — categorical Bellman with guide observations                  #
# --------------------------------------------------------------------------- #

def update_critic(
    guide_actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """
    Categorical distributional Bellman update.

    The critic is always evaluated on guide (clean) observations.
    Next-action bootstrap uses the guide actor so the TD target is as accurate
    as possible (the guide has full state information).
    """
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            # Next actions from guide actor (clean next obs).
            next_actions, next_info = guide_actor(
                observations=batch["guide_next_observation"],
                training=False,
            )
            next_actions       = next_actions.clone()
            next_log_probs     = next_info["log_prob"].clone()

            temp_value         = temperature()
            next_actor_entropy = temp_value * next_log_probs

            # Run target critic on (guide_obs ‖ guide_next_obs) in one batched forward.
            obs_all = torch.cat([batch["guide_observation"], batch["guide_next_observation"]], dim=0)
            act_all = torch.cat([batch["action"], next_actions], dim=0)

            qs_all, q_infos_all = target_critic(
                observations=obs_all,
                actions=act_all,
                training=True,
            )
            next_qs        = qs_all.chunk(2, dim=1)[1]           # (2, B)
            next_q_logprob = q_infos_all["log_prob"].chunk(2, dim=1)[1]  # (2, B, num_bins)
            next_q_logprob = _select_min_q_log_probs(next_qs, next_q_logprob)

            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_logprob,
                reward=batch["guide_reward"],  # shaped: env reward + α·height_gain
                done=batch["terminated"],
                actor_entropy=next_actor_entropy,
                gamma=gamma ** n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()

        # Current-step critic predictions.
        pred_qs_all, pred_q_infos = critic(
            observations=obs_all,
            actions=act_all,
            training=True,
        )
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]  # (2, B, num_bins)

        ce_loss     = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
        critic_loss = ce_loss.mean()

    assert critic.optimizer is not None
    critic.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(critic_loss).backward()
        grad_scaler.step(critic.optimizer)
        grad_scaler.update()
    else:
        critic_loss.backward()
        critic.optimizer.step()

    if critic.scheduler is not None:
        critic.scheduler.step()
    critic.normalize_parameters()

    return _prefix({
        "loss":              critic_loss,
        "max_entropy_bonus": max_entropy_bonus,
    }, "critic")


# --------------------------------------------------------------------------- #
# Shared helpers (unchanged from FlashSAC)                                     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def update_target_network(target_network: Network) -> dict[str, torch.Tensor]:
    target_network.ema_update_parameters()
    return {}


def update_temperature(
    temperature: Network,
    entropy: torch.Tensor,
    target_entropy: float,
) -> dict[str, torch.Tensor]:
    temp_value     = temperature().clone()
    temp_loss      = temp_value * (entropy.detach() - target_entropy).mean()

    assert temperature.optimizer is not None
    temperature.optimizer.zero_grad(set_to_none=True)
    temp_loss.backward()
    temperature.optimizer.step()
    if temperature.scheduler is not None:
        temperature.scheduler.step()

    return _prefix({"value": temp_value, "loss": temp_loss}, "temperature")
