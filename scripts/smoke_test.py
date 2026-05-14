from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pandsvn_drl import DQNConfig, ReplayBuffer, build_dqn
from pandsvn_drl.replay_buffer import Transition


def main() -> None:
    cfg = DQNConfig(num_neighbors=2, hidden_units=32, replay_capacity=100)
    model = build_dqn(cfg.state_dim, cfg.action_dim, cfg.hidden_units)

    state = np.zeros((1, cfg.state_dim), dtype=np.float32)
    q_values = model.predict(state, verbose=0)
    assert q_values.shape == (1, cfg.action_dim)

    buffer = ReplayBuffer(capacity=cfg.replay_capacity)
    buffer.add(Transition(state=state[0], action=0, reward=1.0, next_state=None, done=True))
    assert len(buffer.sample(1)) == 1

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
