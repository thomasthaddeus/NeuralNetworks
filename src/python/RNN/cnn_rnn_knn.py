"""cnn_rnn_knn.py

This module defines the CNNToRNN and RNNToKNN classes, which provide
functionality for combining a Convolutional Neural Network (CNN)
and a Recurrent Neural Network (RNN), and using the output of the CNNToRNN
model to train and make predictions using a K-Nearest
Neighbors (k-NN) classifier.

Classes:
    CNNToRNN: A model that combines a CNN and RNN.
    RNNToKNN: Uses the output of CNNToRNN to train and make predictions using k-NN.

"""

from typing import Tuple
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from rnn import RNN


class CNNToRNN(Model):
    """
    CNNToRNN is a model that combines a Convolutional Neural Network (CNN) and
    a Recurrent Neural Network (RNN).

    Attributes:
        cnn (Sequential): The CNN part of the model.
        rnn (SequentialFeatureSelector): The RNN part of the model.

    Methods:
        __init__(self, image_shape: Tuple[int, int, int], num_classes: int):
        Initializes the CNNToRNN model.
        call(self, inputs: np.ndarray) -> np.ndarray: Performs the forward pass
        of the model.
    """
    def __init__(self, image_shape: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()

        self.cnn = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
        ])

        self.rnn = SequentialFeatureSelector([
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the CNNToRNN model.

        Args:
            inputs (np.ndarray): The input to the model.

        Returns:
            np.ndarray: The output of the model.
        """
        cnn_output = self.cnn(inputs)
        return RNN(cnn_output)


class RNNToKNN:
    """
    RNNToKNN Summary:
        This class uses the output of the CNNToRNN model to train and make
        predictions using a K-Nearest Neighbors (k-NN) classifier.

    Args:
        n_neighbors (int): The number of neighbors to consider in the k-NN
        classifier.
        model (CNNToRNN): The trained CNNToRNN model to extract features.

    Attributes:
        knn (KNeighborsClassifier): The k-NN classifier.
        model (CNNToRNN): The trained CNNToRNN model used to extract features.

    Methods:
        fit(self, trn_x: np.ndarray, trn_y: np.ndarray) -> None:
            Fits the k-NN classifier on the extracted features.

        predict(self, tst_x: np.ndarray) -> np.ndarray:
            Makes predictions using the k-NN classifier.

    Returns:
        None
    """
    def __init__(self, n_neighbors: int, model: CNNToRNN) -> None:
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model: CNNToRNN = model

    def fit(self, trn_x: np.ndarray, trn_y: np.ndarray) -> None:
        """
        Fits the k-NN classifier on the extracted features.

        Args:
            trn_x (np.ndarray): The training data features.
            trn_y (np.ndarray): The training data labels.

        Returns:
            None
        """
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(trn_x)
        # Fit the k-NN model on the features
        self.knn.fit(features, trn_y)


    def predict(self, tst_x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the k-NN classifier.

        Args:
            tst_x (np.ndarray): The test data features.

        Returns:
            np.ndarray: The predicted labels.
        """
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(tst_x)
        # Use the k-NN model to make predictions
        return self.knn.predict(features)