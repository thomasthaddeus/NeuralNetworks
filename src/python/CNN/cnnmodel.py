import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

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
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

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
cnn.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=128)
cnn.evaluate(X_test, y_test)
