"""
Guided Soft Actor-Critic (GSAC)
================================
Two-actor SAC for partially observable environments.

Actors
------
- Control actor  : sees *noisy* observations (what the real robot senses).
                   Deployed at test time.
- Guide actor    : sees *clean/privileged* observations (simulator ground truth).
                   Training-only. Never deployed.

Critic
------
Always trained on clean (guide) observations.  It is a training-only component
so it can use full state information.

Loss functions
--------------
  Control actor:   L_c = SAC_loss(noisy_obs) + λ · L1(a_c − stop_grad(a_g))
  Guide actor:     L_g = SAC_loss(clean_obs)          (standard SAC)
  Critic:          Bellman TD using clean obs + guide actor for next-action bootstrap

The guidance term L1(a_c − a_g) pulls the control actor toward the guide's
actions, teaching it to behave as if it had clean observations.
"""

from functools import partial
from typing import Any, ClassVar, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax.typing import ArrayLike
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from sbx.common.type_aliases import RLTrainState
from sbx.sac.sac import SAC
from sbx.gsac.buffers import GSACReplayBuffer, GSACReplayBufferSamples, GSACReplayBufferSamplesNp
from sbx.gsac.policies import GSACPolicy


class GSAC(SAC):
    """
    Guided Soft Actor-Critic.

    :param policy: Policy class ("MlpPolicy").
    :param env: Gym environment.
    :param guide_observation_space: Clean/privileged observation space for the guide actor.
        If None, the guide sees the same observations as the control actor (no-POMDP mode).
    :param guidance_weight: λ — weight of the L1 guidance loss added to the control actor loss.
    :param get_guide_obs_fn: fn(obs, env) -> guide_obs.
        Required only when guide_observation_space differs from the control obs space.
        Extracts privileged observations from the environment after each step.
    :param kwargs: All remaining SAC hyperparameters.
    """

    policy_aliases: ClassVar[dict[str, type[GSACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": GSACPolicy,
    }
    policy: GSACPolicy

    def __init__(
        self,
        policy,
        env: GymEnv | str,
        guide_observation_space: spaces.Space | None = None,
        guidance_weight: float = 0.5,
        get_guide_obs_fn: Callable | None = None,
        learning_rate: float | Schedule = 3e-4,
        qf_learning_rate: float | None = None,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        param_resets: list[int] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.guide_observation_space = guide_observation_space
        self.guidance_weight = guidance_weight
        self.get_guide_obs_fn = get_guide_obs_fn

        # guide_observation_space must reach GSACPolicy.build() via policy_kwargs.
        # We set a placeholder here; _setup_model() fills in the resolved space.
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.setdefault("guide_observation_space", None)

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_delay=policy_delay,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            n_steps=n_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            param_resets=param_resets,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,  # delay until we patch policy_kwargs below
        )

        if _init_setup_model:
            self._setup_model()

    def _excluded_save_params(self) -> list[str]:
        # get_guide_obs_fn is a closure that may capture the Isaac Lab env
        # (which holds a pxr.Usd.Stage — not picklable). Exclude it from saves.
        return super()._excluded_save_params() + ["get_guide_obs_fn"]

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # Resolve guide obs space: fall back to env obs space if not provided
        resolved = self.guide_observation_space or self.observation_space
        self.guide_observation_space = resolved

        # Inject into policy_kwargs so GSACPolicy.build() can allocate the guide network
        if self.policy_kwargs is None:
            self.policy_kwargs = {}
        self.policy_kwargs["guide_observation_space"] = resolved

        super()._setup_model()

        # If privileged obs differ from control obs, replace the standard replay buffer
        # with GSACReplayBuffer, which stores both observation streams separately.
        if resolved != self.observation_space:
            assert self.get_guide_obs_fn is not None, (
                "guide_observation_space differs from observation_space. "
                "Provide get_guide_obs_fn to extract privileged obs from the env."
            )
            self.replay_buffer = GSACReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                guide_observation_space=resolved,
                device="cpu",
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **(self.replay_buffer_kwargs or {}),
            )

    # ------------------------------------------------------------------
    # Rollout collection — inject guide obs when using privileged space
    # ------------------------------------------------------------------

    def collect_rollouts(self, env, callback, train_freq, replay_buffer,
                         action_noise=None, learning_starts=0, log_interval=None):
        # Same obs space: standard SB3 collection, nothing extra needed
        if not isinstance(replay_buffer, GSACReplayBuffer):
            return super().collect_rollouts(
                env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval
            )

        # Privileged obs: temporarily wrap buffer.add() to also store guide obs.
        # SB3's collect_rollouts calls replay_buffer.add() internally — we intercept
        # that call and inject guide_obs extracted from the env's extras dict.
        #
        # Timing: at the moment _patched_add is called, the env has already stepped.
        # get_guide_obs_fn reads extras["observations"]["critic"] which holds the
        # POST-step (next) state. So we track the previous step's guide obs in a
        # closure variable and shift it forward each call.
        _original_add = replay_buffer.add
        _self = self
        _prev_guide_obs: list = [None]  # [0] = guide obs from the previous step

        def _patched_add(obs, next_obs, action, reward, done, infos):
            # post-step extras → this is guide_next_obs
            guide_next_obs = _self.get_guide_obs_fn(next_obs, _self.env).copy()
            # guide_obs = what extras held before this step (stored last call)
            # On the very first call we have no previous obs, so fall back to next.
            guide_obs = _prev_guide_obs[0] if _prev_guide_obs[0] is not None else guide_next_obs
            _prev_guide_obs[0] = guide_next_obs  # shift forward for next call
            _original_add(obs, next_obs, action, reward, done, infos,
                          guide_obs=guide_obs, guide_next_obs=guide_next_obs)

        replay_buffer.add = _patched_add  # type: ignore[method-assign]
        try:
            result = super().collect_rollouts(
                env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval
            )
        finally:
            replay_buffer.add = _original_add  # always restore
        return result

    # ------------------------------------------------------------------
    # Training loop (called by SB3 every train_freq steps)
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None

        # Sample a flat batch of size (batch_size * gradient_steps,).
        # _train() will split this internally into gradient_steps mini-batches.
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)

        # Update learning rates (scheduled)
        self._update_learning_rate(self.policy.actor_state.opt_state,       learning_rate=self.lr_schedule(self._current_progress_remaining), name="learning_rate_actor")
        self._update_learning_rate(self.policy.guide_actor_state.opt_state,  learning_rate=self.lr_schedule(self._current_progress_remaining), name="learning_rate_guide_actor")
        self._update_learning_rate(self.policy.qf_state.opt_state,           learning_rate=self.initial_qf_learning_rate or self.lr_schedule(self._current_progress_remaining), name="learning_rate_critic")

        self._maybe_reset_params()

        # Convert replay buffer samples to numpy for JAX
        obs      = data.observations.numpy()
        next_obs = data.next_observations.numpy()

        # Guide obs: either from the privileged buffer stream, or reuse control obs
        if isinstance(data, GSACReplayBufferSamples):
            guide_obs      = data.guide_observations.numpy()
            guide_next_obs = data.guide_next_observations.numpy()
        else:
            guide_obs      = obs
            guide_next_obs = next_obs

        # Discounts (per-transition if available, otherwise uniform gamma)
        if hasattr(data, "discounts") and data.discounts is not None:
            discounts = data.discounts.numpy().flatten()
        else:
            discounts = np.full((batch_size * gradient_steps,), self.gamma, dtype=np.float32)

        np_data = GSACReplayBufferSamplesNp(
            observations=obs,
            actions=data.actions.numpy(),
            next_observations=next_obs,
            dones=data.dones.numpy().flatten(),
            rewards=data.rewards.numpy().flatten(),
            discounts=discounts,
            guide_observations=guide_obs,
            guide_next_observations=guide_next_obs,
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.guide_actor_state,
            self.ent_coef_state,
            self.key,
            (actor_loss, guidance_loss, guide_actor_loss, critic_loss, ent_coef_loss, ent_coef),
        ) = self._train(
            self.tau,
            self.target_entropy,
            gradient_steps,
            jnp.float32(self.guidance_weight),
            np_data,
            self.policy_delay,
            (self._n_updates + 1) % self.policy_delay,
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.guide_actor_state,
            self.ent_coef_state,
            self.key,
        )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates",       self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss",       actor_loss.item())
        self.logger.record("train/guidance_loss",    guidance_loss.item())
        self.logger.record("train/guide_actor_loss", guide_actor_loss.item())
        self.logger.record("train/critic_loss",      critic_loss.item())
        self.logger.record("train/ent_coef_loss",    ent_coef_loss.item())
        self.logger.record("train/ent_coef",         ent_coef.item())

    # ------------------------------------------------------------------
    # JAX update steps (all @jax.jit — must be pure functions)
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def update_critic(
        guide_actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        guide_observations: jax.Array,
        actions: jax.Array,
        guide_next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        discounts: jax.Array,
        key: jax.Array,
    ):
        """
        Bellman update for the critic using clean (guide) observations.

        Next-state actions come from the guide actor — it has clean obs so its
        bootstrap is more accurate than using the noisy control actor.
        """
        key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)

        # --- Bellman target ---
        # Sample next action from guide actor (clean obs)
        dist_next = guide_actor_state.apply_fn(guide_actor_state.params, guide_next_observations)
        next_actions  = dist_next.sample(seed=noise_key)
        next_log_prob = dist_next.log_prob(next_actions)

        ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        next_q = qf_state.apply_fn(
            qf_state.target_params, guide_next_observations, next_actions,
            rngs={"dropout": dropout_key_target},
        )
        next_q = jnp.min(next_q, axis=0) - ent_coef * next_log_prob[:, None]
        target_q = rewards[:, None] + (1 - dones[:, None]) * discounts[:, None] * next_q

        # --- MSE loss ---
        def mse_loss(params):
            current_q = qf_state.apply_fn(
                params, guide_observations, actions,
                rngs={"dropout": dropout_key_current},
            )
            return 0.5 * ((target_q - current_q) ** 2).mean(axis=1).sum()

        qf_loss, grads = jax.value_and_grad(mse_loss)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)

        return qf_state, (qf_loss, ent_coef), key

    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: TrainState,
        guide_actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,         # noisy — control actor input
        guide_observations: jax.Array,   # clean  — guide actor input + Q input
        key: jax.Array,
        lam: jax.Array,
    ):
        """
        SAC loss + L1 guidance loss for the control actor.

        The guide actor is evaluated with stop_gradient so its parameters are
        not updated here (it gets its own separate update in update_guide_actor).
        """
        key, dropout_key, noise_key_c, noise_key_g = jax.random.split(key, 4)

        def loss_fn(params):
            # Control actor: noisy obs → action distribution
            dist_c  = actor_state.apply_fn(params, observations)
            a_c     = dist_c.sample(seed=noise_key_c)
            logp_c  = dist_c.log_prob(a_c).reshape(-1, 1)

            # Guide actor: clean obs → reference action (no gradient into guide)
            guide_params = jax.lax.stop_gradient(guide_actor_state.params)
            dist_g = guide_actor_state.apply_fn(guide_params, guide_observations)
            a_g    = jax.lax.stop_gradient(dist_g.sample(seed=noise_key_g))

            # Q evaluated at (clean obs, control action) — critic lives in guide-obs space
            q_vals  = qf_state.apply_fn(qf_state.params, guide_observations, a_c,
                                        rngs={"dropout": dropout_key})
            min_q   = jnp.min(q_vals, axis=0)
            ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            sac_loss      = (ent_coef * logp_c - min_q).mean()
            guidance_loss = jnp.abs(a_c - a_g).mean()
            total_loss    = sac_loss + lam * guidance_loss
            return total_loss, (-logp_c.mean(), guidance_loss)

        (actor_loss, (entropy, guidance_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, qf_state, actor_loss, guidance_loss, key, entropy

    @staticmethod
    @jax.jit
    def update_guide_actor(
        guide_actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        guide_observations: jax.Array,   # clean obs
        key: jax.Array,
    ):
        """
        Standard SAC actor loss for the guide actor using clean observations.
        No guidance term — the guide is trained purely to maximize Q.
        """
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def loss_fn(params):
            dist_g  = guide_actor_state.apply_fn(params, guide_observations)
            a_g     = dist_g.sample(seed=noise_key)
            logp_g  = dist_g.log_prob(a_g).reshape(-1, 1)

            q_vals  = qf_state.apply_fn(qf_state.params, guide_observations, a_g,
                                        rngs={"dropout": dropout_key})
            min_q   = jnp.min(q_vals, axis=0)
            ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            loss = (ent_coef * logp_g - min_q).mean()
            return loss, -logp_g.mean()

        (guide_loss, guide_entropy), grads = jax.value_and_grad(loss_fn, has_aux=True)(guide_actor_state.params)
        guide_actor_state = guide_actor_state.apply_gradients(grads=grads)
        return guide_actor_state, qf_state, guide_loss, key, guide_entropy

    @classmethod
    def update_actors_and_temperature(
        cls,
        actor_state: TrainState,
        guide_actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        guide_observations: jax.Array,
        target_entropy: ArrayLike,
        key: jax.Array,
        lam: jax.Array,
    ):
        """Update control actor, guide actor, and temperature in sequence."""
        # 1. Control actor (SAC + guidance loss)
        actor_state, qf_state, actor_loss, guidance_loss, key, entropy = cls.update_actor(
            actor_state, guide_actor_state, qf_state, ent_coef_state,
            observations, guide_observations, key, lam,
        )
        # 2. Guide actor (standard SAC)
        guide_actor_state, qf_state, guide_loss, key, _ = cls.update_guide_actor(
            guide_actor_state, qf_state, ent_coef_state, guide_observations, key,
        )
        # 3. Temperature (using control actor entropy)
        ent_coef_state, ent_coef_loss = cls.update_temperature(target_entropy, ent_coef_state, entropy)

        return (actor_state, guide_actor_state, qf_state, ent_coef_state,
                actor_loss, guidance_loss, guide_loss, ent_coef_loss, key)

    # ------------------------------------------------------------------
    # Inner training loop (JIT-compiled, runs gradient_steps iterations)
    # ------------------------------------------------------------------

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "policy_delay", "policy_delay_offset"])
    def _train(
        cls,
        tau: float,
        target_entropy: ArrayLike,
        gradient_steps: int,
        lam: jax.Array,
        data: GSACReplayBufferSamplesNp,
        policy_delay: int,
        policy_delay_offset: int,
        qf_state: RLTrainState,
        actor_state: TrainState,
        guide_actor_state: TrainState,
        ent_coef_state: TrainState,
        key: jax.Array,
    ):
        """
        JIT-compiled loop over gradient_steps mini-batch updates.

        Each iteration:
          1. Slice the next mini-batch from the flat data arrays.
          2. Update critic (always).
          3. Soft-update target critic (always).
          4. Update actors + temperature (every policy_delay steps).
        """
        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // gradient_steps

        # Initial carry — all mutable state threaded through fori_loop
        carry = {
            "actor_state":       actor_state,
            "guide_actor_state": guide_actor_state,
            "qf_state":          qf_state,
            "ent_coef_state":    ent_coef_state,
            "key":               key,
            "info": {
                "actor_loss":       jnp.array(0.0),
                "guidance_loss":    jnp.array(0.0),
                "guide_actor_loss": jnp.array(0.0),
                "qf_loss":          jnp.array(0.0),
                "ent_coef_loss":    jnp.array(0.0),
                "ent_coef_value":   jnp.array(0.0),
            },
        }

        def one_update(i: int, carry: dict) -> dict:
            actor_state       = carry["actor_state"]
            guide_actor_state = carry["guide_actor_state"]
            qf_state          = carry["qf_state"]
            ent_coef_state    = carry["ent_coef_state"]
            key               = carry["key"]
            info              = carry["info"]

            # Slice mini-batch i out of the flat data arrays
            sl = lambda arr: jax.lax.dynamic_slice_in_dim(arr, i * batch_size, batch_size)
            batch_obs            = sl(data.observations)
            batch_actions        = sl(data.actions)
            batch_next_obs       = sl(data.next_observations)
            batch_rewards        = sl(data.rewards)
            batch_dones          = sl(data.dones)
            batch_discounts      = sl(data.discounts)
            batch_guide_obs      = sl(data.guide_observations)
            batch_guide_next_obs = sl(data.guide_next_observations)

            # --- Critic update (every step) ---
            (qf_state, (qf_loss, ent_coef_value), key) = cls.update_critic(
                guide_actor_state, qf_state, ent_coef_state,
                batch_guide_obs, batch_actions, batch_guide_next_obs,
                batch_rewards, batch_dones, batch_discounts, key,
            )
            qf_state = cls.soft_update(tau, qf_state)

            # --- Actor + temperature update (every policy_delay steps) ---
            # jax.lax.cond requires both branches to return identical structure.
            # The false branch returns the current states/losses unchanged.
            def do_update(*args):
                return cls.update_actors_and_temperature(*args)

            def skip_update(*args):
                # Same signature as do_update; return existing states/losses unchanged.
                actor_state, guide_actor_state, qf_state, ent_coef_state = args[0], args[1], args[2], args[3]
                key = args[7]
                return (actor_state, guide_actor_state, qf_state, ent_coef_state,
                        info["actor_loss"], info["guidance_loss"], info["guide_actor_loss"],
                        info["ent_coef_loss"], key)

            (actor_state, guide_actor_state, qf_state, ent_coef_state,
             actor_loss, guidance_loss, guide_actor_loss, ent_coef_loss, key) = jax.lax.cond(
                (policy_delay_offset + i) % policy_delay == 0,
                do_update,
                skip_update,
                actor_state, guide_actor_state, qf_state, ent_coef_state,
                batch_obs, batch_guide_obs, target_entropy, key, lam,
            )

            return {
                "actor_state":       actor_state,
                "guide_actor_state": guide_actor_state,
                "qf_state":          qf_state,
                "ent_coef_state":    ent_coef_state,
                "key":               key,
                "info": {
                    "actor_loss":       actor_loss,
                    "guidance_loss":    guidance_loss,
                    "guide_actor_loss": guide_actor_loss,
                    "qf_loss":          qf_loss,
                    "ent_coef_loss":    ent_coef_loss,
                    "ent_coef_value":   ent_coef_value,
                },
            }

        final = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            final["qf_state"],
            final["actor_state"],
            final["guide_actor_state"],
            final["ent_coef_state"],
            final["key"],
            (
                final["info"]["actor_loss"],
                final["info"]["guidance_loss"],
                final["info"]["guide_actor_loss"],
                final["info"]["qf_loss"],
                final["info"]["ent_coef_loss"],
                final["info"]["ent_coef_value"],
            ),
        )
