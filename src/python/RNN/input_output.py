"""input_ouput.py

_summary_

_extended_summary_

Returns:
    _type_: _description_
"""

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

class CNNToRNN(Model):
    """
    CNNToRNN _summary_

    _extended_summary_

    Args:
        Model (_type_): _description_
    """
    def __init__(self, image_shape, num_classes) -> None:
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
        self.rnn = SequentialFeatureSelector([
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs):
        """
        call _summary_

        _extended_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        var_x = self.cnn(inputs)
        return self.rnn(var_X)

class RNNToKNN:
    """
     _summary_

    _extended_summary_
    """
    def __init__(self, n_neighbors, model) -> None:
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model = model

    def fit(self, trn_x, trn_y) -> None:
        """
        fit _summary_

        _extended_summary_

        Args:
            trn_x (_type_): _description_
            trn_y (_type_): _description_
        """
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(trn_x)
        # Fit the k-NN model on the features
        self.knn.fit(features, trn_y)


    def predict(self, tst_x):
        """
        predict _summary_

        _extended_summary_

        Args:
            tst_x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Extract features from the images with the CNN and RNN
        features = self.model.predict(tst_x)
        # Use the k-NN model to make predictions
        return self.knn.predict(features)
