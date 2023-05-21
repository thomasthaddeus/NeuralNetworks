from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from typing import Tuple

class MnistKerasModel:
    """
    A class to build, train, and use a simple dense neural network for the MNIST dataset using Keras.
    """

    def __init__(self) -> None:
        self.model = None

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load the MNIST dataset.
        """
        try:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        except Exception as e:
            raise Exception("An error occurred while loading the MNIST dataset") from e

        # Preprocess data
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2])).astype('float32') / 255
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype('float32') / 255

        return (x_train, y_train), (x_test, y_test)

    def build_model(self, input_shape: int, num_classes: int) -> None:
        """
        Build a simple dense neural network model.
        """
        if not isinstance(input_shape, int) or not isinstance(num_classes, int):
            raise ValueError("input_shape and num_classes should be integers")

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
        self.model.add(Dense(num_classes, activation='softmax'))

    def train(self, x_train: np.ndarray, y_train: np.ndarray, batch_size: int = 128, epochs: int = 5) -> None:
        """
        Train the model.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("x_train and y_train should be numpy arrays")

        y_train = to_categorical(y_train)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the model.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        if not isinstance(x_test, np.ndarray) or not isinstance(y_test, np.ndarray):
            raise ValueError("x_test and y_test should be numpy arrays")

        y_test = to_categorical(y_test)
        return self.model.evaluate(x_test, y_test, verbose=1)[1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of an image.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        if not isinstance(x, np.ndarray):
            raise ValueError("x should be a numpy array")

        return self.model.predict(x)
