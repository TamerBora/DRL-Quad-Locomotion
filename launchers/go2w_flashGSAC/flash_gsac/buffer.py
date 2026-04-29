"""Replay buffer extended to store both control obs and guide (privileged) obs for FlashGSAC."""

from collections import deque
from typing import Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch

from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.buffers.base_buffer import Batch
from flash_rl.types import NDArray


class GuidedTorchUniformBuffer(TorchUniformBuffer):
    """
    TorchUniformBuffer extended for FlashGSAC's two-stream observation model.

    Each transition carries:
      - observation / next_observation  : noisy control observations (what the real robot sees)
      - guide_observation / guide_next_observation : clean/privileged observations (ground truth)

    The n-step return propagation handles guide_next_observation identically to
    next_observation: when an episode ends at step k inside the n-step window,
    guide_next_obs for that env is taken from step k (not from the end of the window).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        guide_obs_dim: int,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
    ):
        # Set before super().__init__() so reset() can allocate the guide tensors.
        self._guide_obs_dim = guide_obs_dim
        super().__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
            device_type,
        )

    def reset(self) -> None:
        super().reset()
        if not hasattr(self, "_guide_obs_dim"):
            return
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()
        self._guide_observations = torch.empty(
            (m, self._guide_obs_dim), dtype=torch.float32, device=self._device, pin_memory=pin
        )
        self._guide_next_observations = torch.empty(
            (m, self._guide_obs_dim), dtype=torch.float32, device=self._device, pin_memory=pin
        )

    def _get_n_step_prev_transition(self) -> Batch:
        # Let the parent compute the n-step reward / terminated / next_observation.
        result = super()._get_n_step_prev_transition()

        # Propagate guide_next_observation through done steps — identical logic to
        # how the parent propagates next_observation.
        curr = self._n_step_transitions[-1]  # newest transition
        n_step_guide_next_obs = curr["guide_next_observation"].clone()

        for n_step_idx in reversed(range(self._n_step - 1)):
            t = self._n_step_transitions[n_step_idx]
            done_mask = t["terminated"].bool() | t["truncated"].bool()
            n_step_guide_next_obs[done_mask] = t["guide_next_observation"][done_mask]

        # guide_observation (current) comes from the oldest transition automatically
        # because `result` IS self._n_step_transitions[0].
        result["guide_next_observation"] = n_step_guide_next_obs
        return result

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append(
            {k: self._to_tensor(v) for k, v in transition.items()}
        )

        if len(self._n_step_transitions) >= self._n_step:
            n_step_prev = cast(dict[str, torch.Tensor], self._get_n_step_prev_transition())

            add_batch_size = len(n_step_prev["observation"])
            end_idx = self._current_idx + add_batch_size

            if end_idx <= self._max_length:
                idxs: Any = slice(self._current_idx, end_idx)
            else:
                idxs = (
                    torch.arange(add_batch_size, device=self._device) + self._current_idx
                ) % self._max_length

            self._observations[idxs]            = n_step_prev["observation"].to(self._observations.dtype)
            self._next_observations[idxs]       = n_step_prev["next_observation"].to(self._next_observations.dtype)
            self._actions[idxs]                 = n_step_prev["action"].to(self._actions.dtype)
            self._rewards[idxs]                 = n_step_prev["reward"].to(self._rewards.dtype)
            self._terminateds[idxs]             = n_step_prev["terminated"].to(self._terminateds.dtype)
            self._truncateds[idxs]              = n_step_prev["truncated"].to(self._truncateds.dtype)
            self._guide_observations[idxs]      = n_step_prev["guide_observation"].to(torch.float32)
            self._guide_next_observations[idxs] = n_step_prev["guide_next_observation"].to(torch.float32)

            self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
            self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
            idxs = torch.randint(0, self._num_in_buffer, (self._sample_batch_size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device)

        return {
            "observation":            self._observations[idxs],
            "action":                 self._actions[idxs],
            "reward":                 self._rewards[idxs],
            "terminated":             self._terminateds[idxs],
            "truncated":              self._truncateds[idxs],
            "next_observation":       self._next_observations[idxs],
            "guide_observation":      self._guide_observations[idxs],
            "guide_next_observation": self._guide_next_observations[idxs],
        }
