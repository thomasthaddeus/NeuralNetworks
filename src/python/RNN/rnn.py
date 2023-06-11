"""rnn.py

This module provides a class for building and training a recurrent neural network (RNN) using Keras from TensorFlow.

Extended Summary:
    The RNN class allows you to easily construct an RNN model with multiple
    layers, specify activation functions, loss functions, and optimizers, and
    train the model on input data.

Returns:
    type: Description of the return value(s).

"""

import tensorflow as tf
from sklearn.model_selection import train_test_split


class RNN:
    def __init__(self, layers, activation_fn, loss_fn, optimizer):
        """
        Initialize the RNN model.

        Args:
            layers (list): List of integers specifying the number of units in
            each RNN layer.
            activation_fn (str): Activation function to use in the RNN layers.
            loss_fn (str): Loss function to use for training the model.
            optimizer (str): Optimizer to use for training the model.
        """
        self.model = tf.keras.models.Sequential()

        for i, units in enumerate(layers):
            if i == 0:
                self.model.add(
                    tf.keras.layers.SimpleRNN(
                        units, activation=activation_fn, return_sequences=True
                    )
                )
            elif i == len(layers) - 1:
                self.model.add(
                    tf.keras.layers.SimpleRNN(units, activation=activation_fn)
                )
            else:
                self.model.add(
                    tf.keras.layers.SimpleRNN(
                        units, activation=activation_fn, return_sequences=True
                    )
                )

        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the RNN model on the given data.

        Args:
            X (array-like): Input data.
            y (array-like): Target data.
            epochs (int): Number of epochs to train the model (default: 10).
            batch_size (int): Batch size for training (default: 32).
            validation_split (float): Fraction of the data to use for
            validation (default: 0.2).
        """
        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

    def predict(self, X):
        """
        Make predictions using the trained RNN model.

        Args:
            X (array-like): Input data for making predictions.

        Returns:
            array-like: Predicted values.
        """
        return self.model.predict(X)
