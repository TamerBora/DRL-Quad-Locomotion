from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import SquashedGaussianActor
from sbx.common.type_aliases import RLTrainState
from sbx.sac.policies import SACPolicy


class GSACPolicy(SACPolicy):
    """
    SAC policy extended with a second guide actor that operates on privileged observations.

    The control actor (actor_state) observes partial/standard observations and is
    what gets deployed at test time. The guide actor (guide_actor_state) observes
    privileged observations during training and provides a supervision signal.

    :param observation_space: Control (partial) observation space.
    :param action_space: Action space.
    :param lr_schedule: Learning rate schedule.
    :param guide_observation_space: Privileged observation space for the guide actor.
    :param guide_net_arch: Network architecture for the guide actor. Defaults to net_arch_pi.
    :param kwargs: Remaining kwargs forwarded to SACPolicy.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        guide_observation_space: spaces.Space,
        guide_net_arch: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.guide_observation_space = guide_observation_space
        self.guide_net_arch = guide_net_arch  # None → resolved in build() to net_arch_pi

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        # Build control actor, critic, and their states via parent.
        # NOTE: super().build() initializes qf_state with self.observation_space (control obs).
        # We immediately re-initialize it below with guide_observation_space because the critic
        # is always evaluated on guide (privileged) observations, not control observations.
        key = super().build(key, lr_schedule, qf_learning_rate)

        key, guide_key, qf_key, dropout_key = jax.random.split(key, 4)

        # Dummy guide observation for parameter initialization
        if isinstance(self.guide_observation_space, spaces.Dict):
            guide_obs = jnp.array(
                [spaces.flatten(self.guide_observation_space, self.guide_observation_space.sample())]
            )
        else:
            guide_obs = jnp.array([self.guide_observation_space.sample()])

        action = jnp.array([self.action_space.sample()])

        # Re-initialize critic with guide obs dimensions.
        # The qf model (self.qf) was already created by super(); only the params change.
        optimizer_class_qf = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate, **self.optimizer_kwargs
        )
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key, "dropout": dropout_key}, guide_obs, action),
            target_params=self.qf.init({"params": qf_key, "dropout": dropout_key}, guide_obs, action),
            tx=optimizer_class_qf,
        )

        # Resolve guide network architecture
        guide_net_arch = self.guide_net_arch if self.guide_net_arch is not None else self.net_arch_pi

        # Create guide actor (same SquashedGaussianActor class, different input dim)
        self.guide_actor = SquashedGaussianActor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=guide_net_arch,
            activation_fn=self.activation_fn,
        )

        optimizer_class_actor = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1), **self.optimizer_kwargs
        )

        self.guide_actor_state = TrainState.create(
            apply_fn=self.guide_actor.apply,
            params=self.guide_actor.init(guide_key, guide_obs),
            tx=optimizer_class_actor,
        )

        self.guide_actor.apply = jax.jit(self.guide_actor.apply)  # type: ignore[method-assign]

        return key
