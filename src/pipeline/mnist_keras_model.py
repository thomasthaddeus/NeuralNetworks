"""mnist_keras_model.py

A module that implements a simple dense neural network model for the MNIST
dataset using Keras.

This module defines the `MnistKerasModel` class, which provides methods to load
the dataset, build the model, train the model, evaluate its performance, and
make predictions.

Usage:
    # Import the module
    from mnist_keras_model import MnistKerasModel

    # Create an instance of MnistKerasModel
    model = MnistKerasModel()

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = model.load_data()

    # Build the model
    model.build_model(input_shape=784, num_classes=10)

    # Train the model
    model.train(x_train, y_train, batch_size=128, epochs=5)

    # Evaluate the model
    accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Make predictions
    predictions = model.predict(x_test[:10])
    print(predictions)

Note:
    This module requires the Keras library to be installed.

Author:
    Thaddeus Thomas
"""

from typing import Tuple

from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from numpy.typing import NDArray


class MnistKerasModel:
    """
    A class to build, train, and use a simple dense neural network for the MNIST dataset using Keras.

    Attributes:
        model (Sequential or None): The Keras Sequential model representing the neural network.

    Methods:
        load_data() -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
            Load the MNIST dataset.

        build_model(input_shape: int, num_classes: int) -> None:
            Build a simple dense neural network model.

        train(x_train: NDArray, y_train: NDArray, batch_size: int = 128, epochs: int = 5) -> None:
            Train the model.

        evaluate(x_test: NDArray, y_test: NDArray) -> float:
            Evaluate the model.

        predict(x: NDArray) -> NDArray:
            Predict the class of an image.

    Raises:
        ImportError: An error occurred while loading the MNIST dataset.
        ValueError: input_shape and num_classes should be integers.
        ValueError: x_train and y_train should be numpy arrays.
        ValueError: x_test and y_test should be numpy arrays.
        ValueError: x should be a numpy array.

    Returns:
        None

    """

    def __init__(self) -> None:
        self.model = None

    def load_data(
        self,
    ) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """
        Load the MNIST dataset.

        Raises:
            ImportError: An error occurred while loading the MNIST dataset.

        Returns:
            A tuple of tuples containing the training data and test data in the format:
            ((x_train, y_train), (x_test, y_test))

        """

        try:
            (x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
        except Exception as err:
            raise ImportError(
                "An error occurred while loading the MNIST dataset"
            ) from err

        # Preprocess data
        x_trn = (
            x_trn.reshape(
                (x_trn.shape[0], x_trn.shape[1] * x_trn.shape[2])
            ).astype("float32")
            / 255
        )
        x_tst = (
            x_tst.reshape(
                (x_tst.shape[0], x_tst.shape[1] * x_tst.shape[2])
            ).astype("float32")
            / 255
        )

        return (x_trn, y_trn), (x_tst, y_tst)

    def build_model(self, input_shape: int, num_classes: int) -> None:
        """
        Build a simple dense neural network model.

        Args:
            input_shape (int): The shape of the input data.
            num_classes (int): The number of classes in the dataset.

        Raises:
            ValueError: input_shape and num_classes should be integers.

        Returns:
            None

        """
        if not isinstance(input_shape, int) or not isinstance(
            num_classes, int
        ):
            raise ValueError("input_shape and num_classes should be integers")

        self.model = Sequential()
        self.model.add(
            Dense(512, activation="relu", input_shape=(input_shape,))
        )
        self.model.add(Dense(num_classes, activation="softmax"))

    def train(
        self,
        x_train: NDArray,
        y_train: NDArray,
        batch_size: int = 128,
        epochs: int = 5,
    ) -> None:
        """
        Train the model.

        Args:
            x_train (NDArray): The training data.
            y_train (NDArray): The training labels.
            batch_size (int): The batch size for training. Default is 128.
            epochs (int): The number of epochs for training. Default is 5.

        Raises:
            ValueError: Model not built yet. Call build_model first.
            ValueError: x_train and y_train should be numpy arrays.

        Returns:
            None

        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        if not isinstance(x_train, NDArray) or not isinstance(
            y_train, NDArray
        ):
            raise ValueError("x_train and y_train should be numpy arrays")

        y_train = to_categorical(y_train)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"],
        )
        self.model.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1
        )

    def evaluate(self, x_test: NDArray, y_test: NDArray) -> float:
        """
        Evaluate the model.

        Args:
            x_test (NDArray): The test data.
            y_test (NDArray): The test labels.

        Raises:
            ValueError: Model not built yet. Call build_model first.
            ValueError: x_test and y_test should be numpy arrays.

        Returns:
            float: The accuracy of the model on the test data.

        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        if not isinstance(x_test, NDArray) or not isinstance(y_test, NDArray):
            raise ValueError("x_test and y_test should be numpy arrays")

        y_test = to_categorical(y_test)
        return self.model.evaluate(x_test, y_test, verbose=1)[1]

    def predict(self, x: NDArray) -> NDArray:
        """
        Predict the class of an image.

        Args:
            x (NDArray): The input data.

        Raises:
            ValueError: Model not built yet. Call build_model first.
            ValueError: x should be a numpy array.

        Returns:
            NDArray: The predicted class probabilities for each input.

        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        if not isinstance(x, NDArray):
            raise ValueError("x should be a numpy array")

        return self.model.predict(x)
