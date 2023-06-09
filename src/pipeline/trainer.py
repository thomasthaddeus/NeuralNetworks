"""trainer.py

Summary:
This module provides the Trainer class for training a model.

Extended Summary:
The Trainer class allows you to train a model using a specified
optimizer and loss function. It utilizes the Adam optimizer and the Categorical
Cross entropy loss function by default. The trainer class provides methods for
performing a single training step (train_step) and for training the model for a
given number of epochs (fit).

Classes:
    Trainer: Allows training a model with a specified optimizer and loss
    function.

Attributes:
model (object): The model to be trained.
optimizer (keras.optimizers.Optimizer): The optimizer to be used for training the model.
loss_fn (keras.losses.Loss): The loss function to be used for training the model.

Methods:
train_step(train_x, train_y): Perform a single training step.
fit(train_x, train_y, epochs): Train the model on the given data for the specified number of epochs.
"""

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
import tensorflow as tf


class Trainer:
    """
    The Trainer class allows you to train a model using a specified
    optimizer and loss function.

    Attributes:
    model (object): The model to be trained.
    optimizer (keras.optimizers.Optimizer): The optimizer to be used for
    training the model.
    loss_fn (keras.losses.Loss): The loss function to be used for training the
    model.
    """

    def __init__(self, model, learning_rate=0.001):
        """
        Initializes the Trainer class with a model, optimizer, and loss function.

        Args:
            model (object): The model to be trained.
            learning_rate (float): Learning rate for the optimizer (default: 0.001).
        """
        self.model = model
        self.optimizer = Adam(learning_rate)
        self.loss_fn = CategoricalCrossentropy()

    def train_step(self, train_x, train_y):
        """
        Perform a single training step.

        Args:
            train_x: Input training data.
            train_y: Target training data.

        Returns:
            float: The loss value for the training step.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(train_x, training=True)
            loss = self.loss_fn(train_y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss

    def fit(self, train_x, train_y, epochs):
        """
        Train the model on the given data for the specified number of epochs.

        Args:
            train_x: Input training data.
            train_y: Target training data.
            epochs (int): Number of epochs to train the model.

        Returns:
            None
        """
        for epoch in range(epochs):
            loss = self.train_step(train_x, train_y)
            print(f"Epoch {epoch + 1}, Loss: {loss}")
