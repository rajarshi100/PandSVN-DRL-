from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import EnvConfig, DQNConfig
from .dqn import DQNAgent
from .environment import VehicularSensingEnv
from .memory import ReplayBuffer


def run_training(args: argparse.Namespace) -> None:
    env_config = EnvConfig(num_neighbors=args.num_neighbors, num_rois=args.num_rois, seed=args.seed)
    dqn_config = DQNConfig(batch_size=args.batch_size, replay_capacity=args.replay_capacity)
    env = VehicularSensingEnv(env_config)
    agent = DQNAgent(env.state_dim, env.num_actions, dqn_config, seed=args.seed)
    replay = ReplayBuffer(args.replay_capacity, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_reward", "steps", "epsilon"])
        writer.writeheader()
        for ep in range(args.episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(args.steps_per_episode):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                replay.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if len(replay) >= args.batch_size:
                    batch = replay.sample(args.batch_size)
                    agent.train_on_batch(batch)
                if done:
                    break
            writer.writerow({"episode": ep, "total_reward": total_reward, "steps": step + 1, "epsilon": agent.epsilon()})
            print(f"episode={ep} reward={total_reward:.2f} steps={step+1} epsilon={agent.epsilon():.3f}")
    print(f"wrote {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DQN agent for the lightweight PandSVN-DRL environment.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--num-neighbors", type=int, default=7)
    parser.add_argument("--num-rois", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
