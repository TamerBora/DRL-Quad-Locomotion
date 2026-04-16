from typing import NamedTuple

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class GSACReplayBufferSamples(NamedTuple):
    """Replay buffer samples with privileged guide observations (torch tensors)."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    guide_observations: torch.Tensor
    guide_next_observations: torch.Tensor


class GSACReplayBufferSamplesNp(NamedTuple):
    """Replay buffer samples converted to numpy — passed into the JIT'd _train loop."""
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    guide_observations: np.ndarray
    guide_next_observations: np.ndarray


class GSACReplayBuffer(ReplayBuffer):
    """
    Replay buffer that stores both control (partial) observations and privileged
    guide observations for the GSAC algorithm.

    :param buffer_size: Maximum number of transitions to store.
    :param observation_space: Control (partial) observation space.
    :param action_space: Action space.
    :param guide_observation_space: Privileged (full) observation space for the guide actor.
    :param device: PyTorch device (kept as 'cpu' for easy numpy conversion).
    :param n_envs: Number of parallel environments.
    :param kwargs: Additional arguments forwarded to ReplayBuffer.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        guide_observation_space: spaces.Space,
        device: str = "cpu",
        n_envs: int = 1,
        **kwargs,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            **kwargs,
        )
        self.guide_observation_space = guide_observation_space
        guide_obs_shape = guide_observation_space.shape

        self.guide_observations = np.zeros(
            (self.buffer_size, self.n_envs, *guide_obs_shape),
            dtype=guide_observation_space.dtype,
        )
        self.guide_next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *guide_obs_shape),
            dtype=guide_observation_space.dtype,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict],
        guide_obs: np.ndarray | None = None,
        guide_next_obs: np.ndarray | None = None,
    ) -> None:
        """
        Add a transition. Falls back to control obs when guide obs not provided.

        :param obs: Control (partial) observation, shape (n_envs, *obs_shape).
        :param next_obs: Control next observation.
        :param action: Action taken.
        :param reward: Reward received.
        :param done: Episode done flag.
        :param infos: Info dicts from env.
        :param guide_obs: Privileged observation, shape (n_envs, *guide_obs_shape).
        :param guide_next_obs: Privileged next observation.
        """
        pos = self.pos  # capture BEFORE super().add() increments it

        if guide_obs is None:
            guide_obs = obs
        if guide_next_obs is None:
            guide_next_obs = next_obs

        self.guide_observations[pos] = guide_obs.reshape(
            (self.n_envs, *self.guide_observation_space.shape)
        )
        self.guide_next_observations[pos] = guide_next_obs.reshape(
            (self.n_envs, *self.guide_observation_space.shape)
        )

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> GSACReplayBufferSamples:
        # Replicate parent's env-index sampling so guide obs use identical indices
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
            guide_next_obs = self.guide_next_observations[
                (batch_inds + 1) % self.buffer_size, env_indices, :
            ]
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )
            guide_next_obs = self.guide_next_observations[batch_inds, env_indices, :]

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            self.guide_observations[batch_inds, env_indices, :],
            guide_next_obs,
        )
        # Standard fields → torch tensors; guide obs also converted via to_torch
        tensors = tuple(map(self.to_torch, data))
        return GSACReplayBufferSamples(*tensors)
