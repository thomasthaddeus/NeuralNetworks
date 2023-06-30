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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from models.knn import KNNModel
from models.cnn import CNNModel
from models.rnn import RNN
from visualization.viz import Visualization


def main():
    """
    main _summary_

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
    knn_predictions = knn_model.predict(test_data)
    cnn_predictions = cnn_model.predict(test_data)
    rnn_predictions = rnn_model.predict(test_data)

    # Initialize Visualization
    vis = Visualization(knn_model)

    # Visualize training history for CNN and RNN
    vis.plot_training_history(rnn_model.history)

    # Visualize confusion matrix for all models
    vis.plot_confusion_matrix(test_labels, knn_predictions)
    vis.plot_confusion_matrix(test_labels, cnn_predictions.argmax(axis=-1))
    vis.plot_confusion_matrix(test_labels, rnn_predictions.argmax(axis=-1))


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
    return train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )


if __name__ == "__main__":
    main()
