from __future__ import annotations

import numpy as np
from .config import EnvConfig


class VehicularSensingEnv:
    """Lightweight gym-style simulator for joint sensing/processing decisions.

    State vector layout:
        [user_queue, neighbor_1_queue, neighbor_1_roi, ..., neighbor_N_queue, neighbor_N_roi]

    Action encoding:
        The integer action is interpreted as a base-3 vector of length N.
        For each neighbour i:
            0 = no sensing request
            1 = request processed result
            2 = request raw sensor data

    This is a compact refactor scaffold, not a full reproduction of every
    stochastic detail in the paper. It preserves the main decision structure:
    location-dependent utility, queue pressure, sensing cost, deadline-sensitive
    processing, and larger action spaces for DQN.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.num_actions = 3 ** self.config.num_neighbors
        self.state_dim = 1 + 2 * self.config.num_neighbors
        self.t = 0
        self.user_queue = 0
        self.neighbor_queues = np.zeros(self.config.num_neighbors, dtype=np.int64)
        self.neighbor_rois = np.zeros(self.config.num_neighbors, dtype=np.int64)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.user_queue = int(self.rng.integers(0, self.config.max_queue_size + 1))
        self.neighbor_queues = self.rng.integers(
            0, self.config.max_queue_size + 1, size=self.config.num_neighbors, dtype=np.int64
        )
        self.neighbor_rois = self.rng.integers(0, self.config.num_rois, size=self.config.num_neighbors, dtype=np.int64)
        return self._state()

    def step(self, action: int):
        action_vec = self.decode_action(action)
        reward = self._compute_reward(action_vec)
        self._advance_queues(action_vec)
        self._move_vehicles()
        self.t += 1
        done = self.t >= self.config.episode_length
        return self._state(), float(reward), done, {}

    def decode_action(self, action: int) -> np.ndarray:
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action must be in [0, {self.num_actions})")
        out = np.zeros(self.config.num_neighbors, dtype=np.int64)
        x = int(action)
        for i in range(self.config.num_neighbors):
            out[i] = x % 3
            x //= 3
        return out

    def encode_action(self, action_vec: np.ndarray) -> int:
        if len(action_vec) != self.config.num_neighbors:
            raise ValueError("action_vec has wrong length")
        value = 0
        scale = 1
        for a in action_vec:
            if a not in (0, 1, 2):
                raise ValueError("each action component must be 0, 1, or 2")
            value += int(a) * scale
            scale *= 3
        return value

    def _state(self) -> np.ndarray:
        pieces = [np.asarray([self.user_queue], dtype=np.float32)]
        for q, r in zip(self.neighbor_queues, self.neighbor_rois):
            pieces.append(np.asarray([q, r], dtype=np.float32))
        state = np.concatenate(pieces)
        # Scale queues and RoI indices for neural network stability.
        state[0::2] = state[0::2] / max(1, self.config.max_queue_size)
        state[2::2] = state[2::2] / max(1, self.config.num_rois - 1)
        return state.astype(np.float32)

    def _compute_reward(self, action_vec: np.ndarray) -> float:
        reward = 0.0
        active = int(np.sum(action_vec != 0))
        sensed_rois = set()
        for i, a in enumerate(action_vec):
            if a == 0:
                continue
            roi = int(self.neighbor_rois[i])
            # Avoid double-counting the same RoI, mirroring the paper's reward structure.
            if roi in sensed_rois:
                continue
            sensed_rois.add(roi)
            base_utility = float(self.config.roi_utilities[roi])
            if a == 1:  # processed result from neighbour
                queue_penalty = self.neighbor_queues[i] / max(1, self.config.max_queue_size)
                success_prob = max(0.0, 1.0 - 0.35 * queue_penalty)
            else:  # raw data processed locally by user
                queue_penalty = self.user_queue / max(1, self.config.max_queue_size)
                success_prob = max(0.0, 1.0 - 0.45 * queue_penalty)
            reward += base_utility * success_prob
        reward -= self.config.sensing_cost * active
        return reward

    def _advance_queues(self, action_vec: np.ndarray) -> None:
        raw_requests = int(np.sum(action_vec == 2))
        processed_requests = (action_vec == 1).astype(np.int64)
        self.user_queue = min(self.config.max_queue_size, self.user_queue + raw_requests)
        self.neighbor_queues = np.minimum(self.config.max_queue_size, self.neighbor_queues + processed_requests)
        self.user_queue = max(0, self.user_queue - self.rng.poisson(self.config.processing_rate_user))
        processed = self.rng.poisson(self.config.processing_rate_neighbor, size=self.config.num_neighbors)
        self.neighbor_queues = np.maximum(0, self.neighbor_queues - processed)

    def _move_vehicles(self) -> None:
        # Simple RoI transition: most vehicles stay nearby, a few move to adjacent/random RoIs.
        mask = self.rng.random(self.config.num_neighbors) < 0.25
        shifts = self.rng.choice([-1, 1], size=self.config.num_neighbors)
        self.neighbor_rois = np.where(mask, (self.neighbor_rois + shifts) % self.config.num_rois, self.neighbor_rois)
