import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib import image
from keras.utils import plot_model


class CNNModel:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        return history

    def plot_model(self):
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return image(filename='model_plot.png')

    def plot_training_history(self, history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')

    def predict(self, X):
        return self.model.predict(X)

# Example usage
cnn = CNNModel()
cnn.build_model(input_shape=(28, 28, 1), num_classes=10)
X_train, y_train, X_test, y_test = cnn.load_data()
history = cnn.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=128)
cnn.evaluate(X_test, y_test)
cnn.plot_model()
cnn.plot_training_history(history)
