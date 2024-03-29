"""rnn.py

This module provides a class for building and training a recurrent neural
network (RNN) using Keras from TensorFlow.

Summary:
    This file defines two classes: CNNToRNN and RNNToKNN. CNNToRNN is a
    subclass of Model from the keras.models module and represents a model that
    combines a Convolutional Neural Network (CNN) and a Recurrent Neural
    Network (RNN).
    `RNNToKNN` is a class that uses the output of `CNNToRNN` to train and make
    predictions using a K-Nearest Neighbors (k-NN)
    classifier.

Extended Summary:
    The RNN class allows you to easily construct an RNN model with multiple
    layers, specify activation functions, loss functions, and optimizers, and
    train the model on input data.

Returns:
    type: Description of the return value(s).
"""

import tensorflow as tf
from keras.layers import LSTM, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier


class RNN:
    """
     _summary_

    _extended_summary_
    """

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

        self.model.compile(
            loss=loss_fn, optimizer=optimizer, metrics=["accuracy"]
        )

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


class CNNToRNN(Model):
    """
    CNNToRNN Summary:
        This class represents a model that combines a Convolutional Neural
        Network (CNN) and a Recurrent Neural Network (RNN).

    Args:
        Model (type): The base class for the CNNToRNN model.

    Attributes:
        cnn (Sequential): The CNN part of the model.
        rnn (SequentialFeatureSelector): The RNN part of the model.

    Methods:
        __init__(self, image_shape, num_classes): Initializes the CNNToRNN
        model with the given image shape and number of classes.
        call(self, inputs): Performs the forward pass of the model.

    Returns:
        None
    """

    def __init__(self, image_shape, num_classes) -> None:
        super().__init__()

        # Define the CNN part
        self.cnn = Sequential(
            [
                Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=image_shape,
                ),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
            ]
        )

        # Define the RNN part
        self.rnn = SequentialFeatureSelector(
            [
                LSTM(128, return_sequences=True),
                LSTM(128),
                Dense(num_classes, activation="softmax"),
            ]
        )

    def call(self, inputs):
        """
        call Summary:
            Performs the forward pass of the CNNToRNN model.

        Args:
            inputs (type): The input to the model.

        Returns:
            type: The output of the model.
        """
        var_x = self.cnn(inputs)
        return self.rnn(var_x)


class RNNToKNN:
    """
    RNNToKNN Summary:
        This class uses the output of the CNNToRNN model to train and make
        predictions using a K-Nearest Neighbors (k-NN) classifier.

    Args:
        n_neighbors (int): The number of neighbors to consider in the k-NN
        classifier.
        model: The trained CNNToRNN model to extract features.

    Attributes:
        knn (KNeighborsClassifier): The k-NN classifier.
        model: The trained CNNToRNN model used to extract features.

    Methods:
        fit(self, trn_x, trn_y): Fits the k-NN classifier on the extracted
        features.
        predict(self, tst_x): Makes predictions using the k-NN classifier.

    Returns:
        None
    """

    def __init__(self, n_neighbors, model):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model = model

    def fit(self, trn_x, trn_y):
        """
        fit Summary:
            Fits the k-NN classifier on the extracted features.

        Args:
            trn_x (type): The training data features.
            trn_y (type): The training data labels.

        Returns:
            None
        """
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(trn_x)
        # Fit the k-NN model on the features
        self.knn.fit(features, trn_y)

    def predict(self, tst_x):
        """
        predict Summary:
            Makes predictions using the k-NN classifier.

        Args:
            tst_x (type): The test data features.

        Returns:
            type: The predicted labels.
        """
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(tst_x)
        # Use the k-NN model to make predictions
        return self.knn.predict(features)
