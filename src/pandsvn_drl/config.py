from dataclasses import dataclass


@dataclass
class DQNConfig:
    """Configuration for the DQN policy network scaffold."""

    num_neighbors: int = 7
    num_regions: int = 8
    max_queue_size: int = 3
    hidden_units: int = 512
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000

    @property
    def state_dim(self) -> int:
        # Paper uses state vector length 2N + 1 for N neighbours.
        return 2 * self.num_neighbors + 1

    @property
    def action_dim(self) -> int:
        # Each neighbouring vehicle has three actions: no sensing,
        # send processed result, or send raw sensor data.
        return 3 ** self.num_neighbors
