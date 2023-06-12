"""main.py

Summary:
    This script is the main driver for a machine learning application. It loads and preprocesses data, initializes various models, trains them, tests them and visualizes the results.

Extended Summary:
    The script begins by loading and preprocessing data using the methods provided in the DataPreparation class. Next, it initializes k-nearest neighbors, convolutional neural network, and recurrent neural network models. The models are then trained on the preprocessed data, tested and their accuracies are calculated. Finally, the Visualization class is used to visualize the model accuracies.
"""

from sklearn import preprocessing
from data_preparation import DataPreparation
from models.knn import KNNModel
from models.cnn import CNNModel
from models.rnn import RNN
from viz import Visualization

def main():
    """Initialize data preparation object"""
    data_prep = DataPreparation(preprocessing, split_ratio)

    # Load and preprocess data
    raw_data = load_data()
    data = data_prep.preprocess(raw_data)
    train_data, test_data, train_labels, test_labels = data_prep.split(data, labels)

    # Initialize models
    knn_model = KNNModel()
    cnn_model = CNNModel()
    rnn_model = RNN()

    # Train models
    knn_model.train(train_data, train_labels)
    cnn_model.train(train_data, train_labels)
    rnn_model.train(train_data, train_labels)

    # Test models
    knn_accuracy = knn_model.test(test_data, test_labels)
    cnn_accuracy = cnn_model.test(test_data, test_labels)
    rnn_accuracy = rnn_model.test(test_data, test_labels)

    # Visualize results
    visualization = Visualization()
    visualization.plot_accuracy(knn_accuracy, cnn_accuracy, rnn_accuracy)

if __name__ == "__main__":
    main()
