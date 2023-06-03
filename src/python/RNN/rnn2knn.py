"""rnn2knn.py
_summary_

_extended_summary_

Returns:
    _type_: _description_
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy.typing import NDArray
from cnn_rnn_knn import CNNToRNN


class RNNToKNN:
    """
    RNNToKNN Summary:
        This class uses the output of the CNNToRNN model to train and make predictions using a K-Nearest Neighbors (k-NN) classifier.

    Args:
        n_neighbors (int): The number of neighbors to consider in the k-NN classifier.
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
