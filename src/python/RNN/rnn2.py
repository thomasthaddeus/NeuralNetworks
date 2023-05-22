"""rnn2.py

This script defines a class for a Recurrent Neural Network (RNN) using Long Short-Term Memory
(LSTM) layers. The RNN is implemented with the TensorFlow library and can be utilized for
sequence prediction tasks, like time-series forecasting or natural language processing tasks.

Extended Summary:

The RNN class has two key methods:
1. `train`: Trains the model on the provided input and target data.
2. `predict`: Makes predictions with the trained model based on the input data.

The model is initialized and structured in the `__init__` method, where LSTM layers are added
according to the number of units specified in the `layers` parameter. The
`return_sequences` parameter is set to `True` for all LSTM layers except the last one.
The model is then compiled with the specified loss function and optimizer.

Usage:
from rnn2 import RNN

model = RNN(layers=[64, 128, 64], activation_fn='tanh', loss_fn='mse', optimizer='adam')
model.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
predictions = model.predict(X_test)

Returns:
RNN class instance which can be used to train and predict sequence data.
"""


import tensorflow as tf


class RNN:
    """
    A class to implement an RNN model using LSTM layers for sequence data.

    Attributes:
        model (Sequential): A Sequential object from Keras to build the model.
        history (History): A Keras History object to record training history.

    Methods:
        train(X, y, epochs, batch_size, validation_split): Train the RNN model on given data.
        predict(X): Predict the outputs for given inputs using the trained model.
    """

    def __init__(self, layers, activation_fn, loss_fn, optimizer) -> None:
        """
        Constructs all the necessary attributes for the RNN object.

        Args:
            layers (list): The number of units in each LSTM layer.
            activation_fn (str): The activation function to use in the LSTM layers.
            loss_fn (str): The loss function to use for the optimization.
            optimizer (str): The optimizer to use for training the model.
        """
        self.model = tf.keras.models.Sequential()

        for i, units in enumerate(layers):
            if i == 0:
                # Add the first LSTM layer with return_sequences=True
                self.model.add(
                    tf.keras.layers.LSTM(
                        units, activation=activation_fn, return_sequences=True
                    )
                )
            elif i == len(layers) - 1:
                # Add the last LSTM layer without return_sequences
                self.model.add(tf.keras.layers.LSTM(units, activation=activation_fn))
            else:
                # Add any other LSTM layers with return_sequences=True
                self.model.add(
                    tf.keras.layers.LSTM(
                        units, activation=activation_fn, return_sequences=True
                    )
                )

        # Compile the model with the given loss function and optimizer
        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

    def train(self, var_x, var_y, epochs=10, batch_size=32, validation_split=0.2):
        """
        Trains the RNN model on the given data.

        Args:
            X (array-like): The input data.
            y (array-like): The target output data.
            epochs (int, optional): The number of epochs to train the model. Defaults to 10.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            validation_split (float, optional): The fraction of the data to use for validation.
                Defaults to 0.2.
        """
        # Train the model and keep the history
        self.history = self.model.fit(
            var_x,
            var_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

    def predict(self, var_x):
        """
        Predicts the outputs for the given inputs using the trained model.

        Args:
            X (array-like): The input data.

        Returns:
            array-like: The predicted outputs.
        """
        # Use the model to make predictions
        return self.model.predict(var_x)
