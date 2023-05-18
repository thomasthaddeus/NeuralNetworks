from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from sklearn.neighbors import KNeighborsClassifier

class CNNToRNN(Model):
    def __init__(self, image_shape, num_classes):
        super().__init__()

        # Define the CNN part
        self.cnn = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
        ])

        # Define the RNN part
        self.rnn = Sequential([
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs):
        x = self.cnn(inputs)
        return self.rnn(x)

class RNNToKNN:
    def __init__(self, n_neighbors, model):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model = model

    def fit(self, X_train, y_train):
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(X_train)

        # Fit the k-NN model on the features
        self.knn.fit(features, y_train)

    def predict(self, X_test):
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(X_test)

        # Use the k-NN model to make predictions
        return self.knn.predict(features)
