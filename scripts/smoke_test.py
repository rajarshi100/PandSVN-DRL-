from pandsvn_drl import EnvConfig, VehicularSensingEnv, ReplayBuffer


def main() -> None:
    env = VehicularSensingEnv(EnvConfig(seed=123))
    state = env.reset()
    assert state.shape[0] == env.state_dim
    action = env.encode_action([0] * env.config.num_neighbors)
    next_state, reward, done, _ = env.step(action)
    buffer = ReplayBuffer(capacity=10, seed=123)
    buffer.add(state, action, reward, next_state, done)
    batch = buffer.sample(1)
    assert batch.states.shape[0] == 1
    print("Smoke test passed.")
    print(f"state_dim={env.state_dim}, num_actions={env.num_actions}, reward={reward:.2f}")


if __name__ == "__main__":
    main()
