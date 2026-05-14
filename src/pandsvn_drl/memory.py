from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random
import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Simple bounded replay buffer for DQN training."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: List[Transition] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        if len(self._data) >= self.capacity:
            self._data.pop(0)
        self._data.append((state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def sample(self, batch_size: int) -> ReplayBatch:
        if not self._data:
            raise ValueError("cannot sample from an empty replay buffer")
        batch_size = min(batch_size, len(self._data))
        samples = self._rng.sample(self._data, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return ReplayBatch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
