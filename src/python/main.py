"""main.py

Summary:
    This script is the main driver for a machine learning application. It loads
    and preprocesses data, initializes various models, trains them, tests them
    and visualizes the results.

Extended Summary:
    The script begins by loading and preprocessing data using the methods
    provided in the DataPreparation class. Next, it initializes k-nearest
    neighbors, convolutional neural network, and recurrent neural network
    models. The models are then trained on the preprocessed data, tested and
    their accuracies are calculated. Finally, the Visualization class is used
    to visualize the model accuracies.
"""

# Add the necessary imports
from data_preparation import DataPreparation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from models.knn import KNNModel
from models.cnn import CNNModel
from models.rnn import RNN
from visualization.viz import Visualization


def main():
    """
    main _summary_
    # Initialize data preparation object
    data_prep = DataPreparation(preprocessing_fn=preprocessing.scale, split_ratio=0.2)

    _extended_summary_
    """
    # Load and preprocess data
    data, labels = load_data()
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # Preprocess the data if needed
    # scaler = preprocessing.StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)

    # Initialize models
    knn_model = KNNModel(k=3, problem_type='classification')
    cnn_model = CNNModel()
    cnn_model.build_model(28, 28, 1)
    rnn_model = RNN(
        layers=[100, 50],
        activation_fn="relu",
        loss_fn="binary_crossentropy",
        optimizer="adam",
    )

    # Train models
    knn_model.train(train_data, train_labels)
    cnn_model.train(
        train_data, train_labels,
        test_data, test_labels,
        epochs=50, batch_size=32
    )
    rnn_model.train(train_data, train_labels, epochs=50, batch_size=32)

    # Test models
    knn_accuracy = knn_model.test(test_data, test_labels)
    cnn_accuracy = cnn_model.test(test_data, test_labels)
    rnn_accuracy = rnn_model.test(test_data, test_labels)


if __name__ == "__main__":
    main()
