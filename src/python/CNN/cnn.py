"""cnn_model.py

This module defines the CNNModel class for building and training a Convolutional
Neural Network (CNN) model for image classification.

Usage:
    cnn = CNNModel()
    cnn.build_model(input_shape=(28, 28, 1), num_classes=10)
    x_train, y_train, x_test, y_test = cnn.load_data()
    cnn.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
    cnn.evaluate(x_test, y_test)
"""

from types import Any, ndarray, dtype
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

class CNNModel:
    """Convolutional Neural Network (CNN) model for image classification."""

    def __init__(self):
        self.model = None

    def build_model(self, input_shape, num_classes):
        """
        Build the CNN model architecture.

        Args:
            input_shape (tuple): Shape of the input data (e.g., (28, 28, 1)).
            num_classes (int): Number of classes in the classification task.
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def load_data(self) -> tuple[ndarray[Any, dtype[Any]],
                               + ndarray[Any, dtype[Any]],
                               + ndarray[Any, dtype[Any]],
                               + ndarray[Any, dtype[Any]]]:
        """
        Load and preprocess the MNIST dataset.

        Returns:
            tuple: Four arrays representing the preprocessed training and test data and labels.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train: ndarray[Any, dtype[Any]] = np.expand_dims(x_train, axis=-1)
        x_test: ndarray[Any, dtype[Any]] = np.expand_dims(x_test, axis=-1)
        y_train: ndarray[Any, dtype[Any]] = to_categorical(y_train, num_classes=10)
        y_test: ndarray[Any, dtype[Any]] = to_categorical(y_test, num_classes=10)
        return x_train, y_train, x_test, y_test


    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        """
        Train the CNN model.

        Args:
            x_train (ndarray): Training data.
            y_train (ndarray): Training labels.
            x_test (ndarray): Test data.
            y_test (ndarray): Test labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))


    def evaluate(self, x_test, y_test):
        """
        Evaluate the trained CNN model on the test data.

        Args:
            x_test (ndarray): Test data.
            y_test (ndarray): Test labels.
        """
        loss: float, accuracy: float = self.model.evaluate(self.x_test, self.y_test)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')

    def predict(self, x):
        """
        Make predictions using the trained CNN model.

        Args:
            x (ndarray): Input data for making predictions.

        Returns:
            ndarray: Predicted probabilities for each class.
        """
        return self.model.predict(x)
