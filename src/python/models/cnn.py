"""cnn.py

This module defines the CNNModel class for building and training a Convolutional
Neural Network (CNN) model for image classification.

Usage:
    cnn = CNNModel()
    cnn.build_model(input_shape=(28, 28, 1), num_classes=10)
    x_train, y_train, x_test, y_test = cnn.load_data()
    cnn.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
    cnn.evaluate(x_test, y_test)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
from matplotlib import image


class CNNModel:
    """Convolutional Neural Network (CNN) model for image classification."""

    def __init__(self) -> None:
        self.model = None

    def build_model(self, input_shape, num_classes):
        """
        Build the CNN model architecture.

        Args:
            input_shape (tuple): Shape of the input data (e.g., (28, 28, 1)).
            num_classes (int): Number of classes in the classification task.
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(num_classes, activation="softmax"))
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def load_data(self):
        """
        Load and preprocess the MNIST dataset.

        Returns:
            tuple: Four arrays representing the preprocessed training and test
            data and labels.
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
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
        if self.model is None:
            raise ValueError(
                "The model is not built yet." "Please call build_model before training."
            )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
        )
        return history

    def evaluate(self, X_test, y_test) -> None:
        """
        Evaluate the trained CNN model on the test data.

        Args:
            x_test (ndarray): Test data.
            y_test (ndarray): Test labels.
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")


    def predict(self, X):
        """
        Make predictions using the trained CNN model.

        Args:
            x (ndarray): Input data for making predictions.

        Returns:
            ndarray: Predicted probabilities for each class.
        """
        return self.model.predict(X)


    def plot_model(self):
        """plot model"""
        plot_model(
            self.model,
            to_file="model_plot.png",
            show_shapes=True,
            show_layer_names=True,
        )
        return image(filename="model_plot.png")


    def plot_training_history(self, history):
        """Plot training & validation accuracy values"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")

        plt.tight_layout()
        plt.show()

def cnn_main():
    """defines a cnnmodel"""
    cnn = CNNModel()
    cnn.build_model(input_shape=(28, 28, 1), num_classes=10)
    X_train, y_train, X_test, y_test = cnn.load_data()
    history = cnn.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=10,
        batch_size=128
    )
    cnn.evaluate(X_test, y_test)
    cnn.plot_model()
    cnn.plot_training_history(history)
