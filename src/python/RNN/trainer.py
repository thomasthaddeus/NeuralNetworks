"""
_summary_

_extended_summary_

Returns:
    _type_: _description_
"""


from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = Adam(learning_rate)
        self.loss_fn = CategoricalCrossentropy()

    def train_step(self, X_train, y_train):
        with tf.GradientTape() as tape:
            predictions = self.model(X_train, training=True)
            loss = self.loss_fn(y_train, predictions)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss

    def fit(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            loss = self.train_step(X_train, y_train)
            print(f"Epoch {epoch+1}, Loss: {loss}")
