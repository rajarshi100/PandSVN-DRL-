from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    steps: int
    epsilon: float
