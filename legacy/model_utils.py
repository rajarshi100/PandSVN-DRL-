import tensorflow as tf


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states          #Input Size
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._model = self._define_model()

    def _define_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(1, self._num_states)),
            tf.keras.layers.Dense(self._num_actions, activation=None)
        ])
        model.compile(loss='mean_squared_error', optimizer='Adam_optimizer')
        return model

    def predict_batch(self, x):                 #Input samples in a batch as a list of numpy arrays
        return self._model.predict_on_batch(x)  #Outputs a numpy array of predictions

    def train_batch(self, x, y):      #x,y : lists of numpy arrays
        train_on_batch(x, y, sample_weight=None, class_weight=None)

