"""rnn.py

using keras from tensorflow

_extended_summary_

Returns:
_type_: _description_
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split

class RNN:
    def __init__(self, layers, activation_fn, loss_fn, optimizer):
        self.model = tf.keras.models.Sequential()

        for i, units in enumerate(layers):
            if i == 0:
                self.model.add(tf.keras.layers.SimpleRNN(units, activation=activation_fn, return_sequences=True))
            elif i == len(layers) - 1:
                self.model.add(tf.keras.layers.SimpleRNN(units, activation=activation_fn))
            else:
                self.model.add(tf.keras.layers.SimpleRNN(units, activation=activation_fn, return_sequences=True))

        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        return self.model.predict(X)
