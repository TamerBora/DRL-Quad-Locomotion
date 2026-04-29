"""
FlashGSACAgent — FlashSAC back-end with Guided SAC two-actor extension.

Architecture
------------
Control actor   : observes noisy/partial observations (what the real robot senses).
                  This is the only actor deployed at test time.
Guide actor     : observes clean/privileged observations (simulator ground truth).
                  Training-only. Provides reference actions for the guidance term.
Critic          : trained on clean (guide) observations using the guide actor for
                  next-action bootstrap.  Uses the categorical distributional head
                  from FlashSAC (C51-style, num_bins bins).

Loss functions
--------------
  Control actor : SAC_loss(noisy_obs, Q(guide_obs, a_c)) + λ·L1(a_c − stop_grad(a_g))
  Guide actor   : standard SAC_loss(clean_obs)
  Critic        : categorical Bellman TD with guide obs throughout
"""

import math
import os
from dataclasses import dataclass, replace, field
from typing import Any, MutableMapping, Optional, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.agents.flashSAC.agent import (
    FlashSACConfig,
    _build_truncated_zeta_cdf,
    _sample_integer_from_cdf,
    _resolve_compile_mode,
    _sample_flashsac_actions,
)
from flash_rl.agents.flashSAC.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
from flash_rl.types import NDArray, Tensor

from flash_gsac.buffer import GuidedTorchUniformBuffer
from flash_gsac.update import (
    update_actor,
    update_guide_actor,
    update_critic,
    update_target_network,
    update_temperature,
)


@dataclass
class FlashGSACConfig(FlashSACConfig):
    # Guidance
    guidance_weight: float = 0.5        # λ: weight of the L1 guidance loss

    # Guide actor architecture (defaults to same size as control actor)
    guide_actor_num_blocks: int = 2
    guide_actor_hidden_dim: int = 128


def _make_network(
    net: torch.nn.Module,
    lr_peak: float,
    lr_lambda,
    device: torch.device,
    use_compile: bool,
    compile_mode: str,
    use_weight_norm: bool,
    use_fused: bool,
    ema_source: Optional[Network] = None,
    ema_tau: Optional[float] = None,
) -> Network:
    """Construct a Network bundle (actor, critic, or temperature) with its optimizer."""
    optimizer = optim.Adam(net.parameters(), lr=lr_peak, fused=use_fused)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step) / lr_peak
    )
    return Network(
        network=net,
        optimizer=optimizer,
        scheduler=scheduler,
        compile_network=use_compile,
        compile_mode=compile_mode,
        use_weight_normalization=use_weight_norm,
        ema_source=ema_source,
        ema_tau=ema_tau,
    )


class FlashGSACAgent(BaseAgent[FlashGSACConfig]):
    """
    FlashGSAC: Guided SAC with the FlashSAC distributional-critic back-end.

    Parameters
    ----------
    observation_space       : gym.spaces.Box  — control (noisy) obs from the env.
    action_space            : gym.spaces.Box
    guide_observation_space : gym.spaces.Box  — clean/privileged obs (critic obs group).
    env_info                : dict from env.reset()
    cfg                     : FlashGSACConfig
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        guide_observation_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: FlashGSACConfig,
    ):
        # ── Observation / action dimensions ─────────────────────────────────
        self._control_obs_dim: int = observation_space.shape[-1]     # type: ignore
        self._guide_obs_dim:   int = guide_observation_space.shape[-1]  # type: ignore
        self._action_dim:      int = action_space.shape[-1]           # type: ignore

        # Control actor obs dim (possibly a subset when asymmetric_observation=True)
        if cfg.asymmetric_observation:
            self._actor_observation_dim: int = env_info["actor_observation_size"][-1]
        else:
            self._actor_observation_dim = self._control_obs_dim

        # Critic lives in guide-obs space
        self._critic_observation_dim: int = self._guide_obs_dim

        # Compute target entropy from action dimension
        temp_target_entropy = 0.5 * self._action_dim * math.log(
            2 * math.pi * math.e * cfg.temp_target_sigma ** 2
        )
        compile_mode = _resolve_compile_mode(cfg.compile_mode)
        cfg = replace(cfg, temp_target_entropy=temp_target_entropy, compile_mode=compile_mode)

        super().__init__(observation_space, action_space, env_info, cfg)
        self._cfg = cfg

        device_str = cfg.device_type
        device_str = (
            device_str
            if device_str.startswith("cuda") and ":" in device_str
            else ("cuda:0" if device_str.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_str)
        use_fused = self._device.type == "cuda" and torch.cuda.is_available()

        # ── Learning rate schedule ───────────────────────────────────────────
        warmup_cosine_lr = warmup_cosine_decay_scheduler(
            init_value=cfg.learning_rate_init,
            peak_value=cfg.learning_rate_peak,
            end_value=cfg.learning_rate_end,
            warmup_steps=cfg.learning_rate_warmup_step,
            decay_steps=cfg.learning_rate_decay_step,
        )
        lr_peak = cfg.learning_rate_peak

        # ── Networks ────────────────────────────────────────────────────────
        # Control actor (noisy obs)
        actor_net = FlashSACActor(
            num_blocks=cfg.actor_num_blocks,
            input_dim=self._actor_observation_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=self._action_dim,
        ).to(self._device)
        self._actor = _make_network(
            actor_net, lr_peak, warmup_cosine_lr, self._device,
            cfg.use_compile, compile_mode, True, use_fused,
        )
        if cfg.use_compile:
            self._actor.network.get_mean_and_std = torch.compile(  # type: ignore
                self._actor.network.get_mean_and_std, mode=compile_mode
            )

        # Guide actor (clean obs)
        guide_actor_net = FlashSACActor(
            num_blocks=cfg.guide_actor_num_blocks,
            input_dim=self._guide_obs_dim,
            hidden_dim=cfg.guide_actor_hidden_dim,
            action_dim=self._action_dim,
        ).to(self._device)
        self._guide_actor = _make_network(
            guide_actor_net, lr_peak, warmup_cosine_lr, self._device,
            cfg.use_compile, compile_mode, True, use_fused,
        )
        if cfg.use_compile:
            self._guide_actor.network.get_mean_and_std = torch.compile(  # type: ignore
                self._guide_actor.network.get_mean_and_std, mode=compile_mode
            )

        # Critic (guide-obs space, input_dim = guide_obs_dim + action_dim)
        critic_net = FlashSACDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            input_dim=self._critic_observation_dim + self._action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            num_bins=cfg.critic_num_bins,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
        ).to(self._device)
        self._critic = _make_network(
            critic_net, lr_peak, warmup_cosine_lr, self._device,
            cfg.use_compile, compile_mode, True, use_fused,
        )

        # Target critic — EMA copy, no optimizer
        target_critic_net = FlashSACDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            input_dim=self._critic_observation_dim + self._action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            num_bins=cfg.critic_num_bins,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
        ).to(self._device)
        target_critic_net.load_state_dict(critic_net.state_dict())
        self._target_critic = Network(
            network=target_critic_net,
            optimizer=None,
            scheduler=None,
            compile_network=cfg.use_compile,
            compile_mode=compile_mode,
            use_weight_normalization=True,
            ema_source=self._critic,
            ema_tau=cfg.critic_target_update_tau,
        )

        # Temperature
        temp_net = FlashSACTemperature(cfg.temp_initial_value).to(self._device)
        self._temperature = _make_network(
            temp_net, lr_peak, warmup_cosine_lr, self._device,
            cfg.use_compile, compile_mode, False, use_fused,
        )

        # Normalise initial parameters
        self._actor.normalize_parameters()
        self._guide_actor.normalize_parameters()
        self._critic.normalize_parameters()
        self._target_critic.normalize_parameters()

        # ── AMP grad scaler ──────────────────────────────────────────────────
        self._grad_scaler = GradScaler(device=self._device.type, enabled=cfg.use_amp)

        # ── Noise repeat (zeta distribution, for exploration) ────────────────
        self._zeta_cdf = _build_truncated_zeta_cdf(
            mu=cfg.actor_noise_zeta_mu, max_n=cfg.actor_noise_zeta_max
        ).to(self._device)
        self._cur_noise_repeat_n     = torch.tensor(1, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._device)
        action_shape = tuple(action_space.shape) if action_space.shape is not None else ()
        self._cached_noise = torch.randn(action_shape, device=self._device)

        # ── Reward normaliser ────────────────────────────────────────────────
        self.reward_normalizer: Optional[RewardNormalizer] = None
        if cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=cfg.gamma,
                G_max=cfg.normalized_G_max,
                load_rms=cfg.load_reward_normalizer,
                device=self._device,
            )

        # ── Replay buffer (stores both control and guide observations) ────────
        self._replay_buffer = GuidedTorchUniformBuffer(
            observation_space=observation_space,
            action_space=action_space,
            guide_obs_dim=self._guide_obs_dim,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
            max_length=cfg.buffer_max_length,
            min_length=cfg.buffer_min_length,
            sample_batch_size=cfg.sample_batch_size,
            device_type=cfg.buffer_device_type,
        )

        self._update_step = 0

    # ---------------------------------------------------------------------- #
    # Interaction                                                              #
    # ---------------------------------------------------------------------- #

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        """Control actor only — guide actor is never deployed."""
        temperature = 1.0 if training else 0.0

        observations = prev_transition["next_observation"]
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]

        observations = torch.as_tensor(observations, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )

        return actions.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        """Add transition to the guided replay buffer and update reward stats.

        The transition dict must contain:
          "observation", "next_observation", "action", "reward",
          "terminated", "truncated",
          "guide_observation", "guide_next_observation".
        """
        self._replay_buffer.add(transition)

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            self.reward_normalizer.update_reward_stats(
                reward=torch.as_tensor(transition["reward"], device=self._device),
                terminated=torch.as_tensor(transition["terminated"], device=self._device),
                truncated=torch.as_tensor(transition["truncated"], device=self._device),
            )

    def can_start_training(self) -> bool:
        return self._replay_buffer.can_sample()

    # ---------------------------------------------------------------------- #
    # Update                                                                   #
    # ---------------------------------------------------------------------- #

    def update(self) -> dict[str, Any]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)

        # Build control actor obs (possibly a subset of the full control obs)
        if self._cfg.asymmetric_observation:
            batch["actor_observation"]      = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, : self._actor_observation_dim]
        else:
            batch["actor_observation"]      = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        do_actor_update = (self._update_step % self._cfg.actor_update_period == 0)

        # Update control actor first (uses current guide actor as reference),
        # then update guide actor — matches original GSAC update ordering.
        if do_actor_update:
            actor_info = update_actor(
                actor=self._actor,
                guide_actor=self._guide_actor,
                critic=self._critic,
                temperature=self._temperature,
                batch=batch,
                guidance_weight=self._cfg.guidance_weight,
                bc_alpha=self._cfg.actor_bc_alpha,
                device=self._device,
                use_amp=self._cfg.use_amp,
                grad_scaler=self._grad_scaler,
            )
            guide_actor_info = update_guide_actor(
                guide_actor=self._guide_actor,
                critic=self._critic,
                temperature=self._temperature,
                batch=batch,
                device=self._device,
                use_amp=self._cfg.use_amp,
                grad_scaler=self._grad_scaler,
            )
            temperature_info = update_temperature(
                temperature=self._temperature,
                entropy=actor_info["actor/entropy"],
                target_entropy=self._cfg.temp_target_entropy,
            )
        else:
            actor_info       = {}
            guide_actor_info = {}
            temperature_info = {}

        critic_info = update_critic(
            guide_actor=self._guide_actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            min_v=self._cfg.critic_min_v,
            max_v=self._cfg.critic_max_v,
            num_bins=self._cfg.critic_num_bins,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            device=self._device,
            use_amp=self._cfg.use_amp,
            grad_scaler=self._grad_scaler,
        )

        update_target_network(self._target_critic)

        self._update_step += 1

        # Flatten all info dicts and convert tensors to floats
        raw: dict[str, Any] = {
            **actor_info,
            **guide_actor_info,
            **critic_info,
            **temperature_info,
        }
        update_info: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)

        return update_info

    # ---------------------------------------------------------------------- #
    # Checkpoint                                                               #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt"))
        self._guide_actor.save(os.path.join(path, "guide_actor.pt"))
        self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt"))
        self._temperature.save(os.path.join(path, "temperature.pt"))
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))
        torch.save(
            {
                "update_step":             self._update_step,
                "grad_scaler_state_dict":  self._grad_scaler.state_dict(),
            },
            os.path.join(path, "agent_state.pt"),
        )
        print(f"\033[32m[FlashGSAC]\033[0m Saved checkpoint {self._update_step} → {path}")

    def save_replay_buffer(self, path: str) -> None:
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashGSAC]\033[0m Saved replay buffer → {path}")

    def load(self, path: str) -> None:
        load_opt = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"),               load_optimizer=load_opt)
        self._guide_actor.load(os.path.join(path, "guide_actor.pt"),   load_optimizer=load_opt)
        self._critic.load(os.path.join(path, "critic.pt"),             load_optimizer=load_opt)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False)
        self._temperature.load(os.path.join(path, "temperature.pt"),   load_optimizer=load_opt)

        if load_opt:
            state = torch.load(os.path.join(path, "agent_state.pt"), map_location=self._device)
            self._update_step = state["update_step"]
            self._grad_scaler.load_state_dict(state["grad_scaler_state_dict"])

        if self._cfg.load_reward_normalizer and self.reward_normalizer is not None:
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))

        print(f"\033[32m[FlashGSAC]\033[0m Loaded checkpoint from {path}")

    def load_replay_buffer(self, path: str) -> None:
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashGSAC]\033[0m Loaded replay buffer from {path}")

    def get_metrics(self) -> dict[str, Any]:
        return {}
