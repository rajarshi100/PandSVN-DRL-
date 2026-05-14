from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import DQNConfig
from .memory import ReplayBatch


class DQNAgent:
    """TensorFlow 2 DQN agent for discrete joint sensing/processing actions."""

    def __init__(self, state_dim: int, num_actions: int, config: DQNConfig | None = None, seed: int = 0) -> None:
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)
        self.online = self._build_model()
        self.target = self._build_model()
        self.update_target()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.train_steps = 0

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.state_dim,), name="state")
        x = tf.keras.layers.Dense(self.config.hidden_units, activation="relu")(inputs)
        x = tf.keras.layers.Dense(self.config.hidden_units, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.num_actions, activation=None, name="q_values")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def update_target(self) -> None:
        self.target.set_weights(self.online.get_weights())

    def epsilon(self) -> float:
        frac = min(1.0, self.train_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.num_actions))
        q_values = self.online(np.asarray(state, dtype=np.float32)[None, :], training=False).numpy()[0]
        return int(np.argmax(q_values))

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        next_q = self.target(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + self.config.gamma * (1.0 - dones) * max_next_q
        with tf.GradientTape() as tape:
            q = self.online(states, training=True)
            action_mask = tf.one_hot(actions, self.num_actions)
            chosen_q = tf.reduce_sum(q * action_mask, axis=1)
            loss = tf.reduce_mean(tf.square(targets - chosen_q))
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))
        return loss

    def train_on_batch(self, batch: ReplayBatch) -> float:
        loss = self._train_step(
            tf.convert_to_tensor(batch.states, dtype=tf.float32),
            tf.convert_to_tensor(batch.actions, dtype=tf.int32),
            tf.convert_to_tensor(batch.rewards, dtype=tf.float32),
            tf.convert_to_tensor(batch.next_states, dtype=tf.float32),
            tf.convert_to_tensor(batch.dones, dtype=tf.float32),
        )
        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.update_target()
        return float(loss.numpy())
