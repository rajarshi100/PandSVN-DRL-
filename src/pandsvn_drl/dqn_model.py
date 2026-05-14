import tensorflow as tf


def build_dqn(state_dim: int, action_dim: int, hidden_units: int = 512) -> tf.keras.Model:
    """Build a feed-forward DQN approximator.

    The IEEE TIV paper describes a DQN with input dimension ``2N + 1``
    and an output node for each joint action. This helper provides a clean
    TensorFlow/Keras implementation of that network scaffold.
    """
    inputs = tf.keras.Input(shape=(state_dim,), name="state")
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(action_dim, activation=None, name="q_values")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dqn_joint_sensing_processing")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model
