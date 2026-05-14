from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    """Configuration for the lightweight vehicular sensing environment."""

    num_neighbors: int = 7
    num_rois: int = 8
    max_queue_size: int = 3
    episode_length: int = 600
    deadline_s: float = 1.5
    processing_rate_user: float = 2.0
    processing_rate_neighbor: float = 1.0
    sensing_cost: float = 20.0
    roi_utilities: Tuple[float, ...] = (1600, 1200, 800, 400, 200, 400, 800, 1200)
    seed: int = 0


@dataclass
class DQNConfig:
    """Configuration for the TensorFlow DQN agent."""

    hidden_units: int = 512
    learning_rate: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    replay_capacity: int = 50000
    batch_size: int = 64
    target_update_interval: int = 500
