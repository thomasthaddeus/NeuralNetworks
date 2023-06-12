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
    # Initialize data preparation object
    data_prep = DataPreparation(preprocessing_fn=preprocessing.scale, split_ratio=0.2)

    # Load and preprocess data
    raw_data = load_data()  # Implement the load_data() function
    data = data_prep.preprocess(raw_data)
    train_data, test_data, train_labels, test_labels = data_prep.split(data, labels)  # Define and assign the 'labels' variable

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

def load_data():
    """
    Load and return the iris dataset.

    Returns:
        tuple: The dataset features and labels.
    """
    iris = load_iris()
    return iris.data, iris.target

def split(data, labels, test_size=0.2, random_state=42):
    """
    Split the data and labels into training and testing sets.

    Args:
        data (array-like): The data to be split.
        labels (array-like): The corresponding labels.
        test_size (float): The fraction of the data to be used for testing (default: 0.2).
        random_state (int): The random seed for reproducible results (default: 42).

    Returns:
        tuple: The training and testing data and labels.
    """
    return train_test_split(data, labels, test_size=test_size, random_state=random_state)





if __name__ == "__main__":
    main()
