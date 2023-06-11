"""main.py

_summary_

_extended_summary_
"""


from config import DataPreparation as data_prep
from knn_main import knn_main
from cnn import CNNModel
from rnn import RNN
from visualization.viz import Visualization

def main():
    # Load and preprocess data
    data_prep.load_data()
    data_prep.preprocess_data()

    # Initialize models
    knn_model = knn_main()
    cnn_model = CNNModel()
    rnn_model = RNN()

    # Train models
    knn_model.train(data_prep.train_data, data_prep.train_labels)
    cnn_model.train(data_prep.train_data, data_prep.train_labels)
    rnn_model.train(data_prep.train_data, data_prep.train_labels)

    # Test models
    knn_accuracy = knn_model.test(data_prep.test_data, data_prep.test_labels)
    cnn_accuracy = cnn_model.test(data_prep.test_data, data_prep.test_labels)
    rnn_accuracy = rnn_model.test(data_prep.test_data, data_prep.test_labels)

    # Visualize results
    visualization = Visualization()
    visualization.plot_accuracy(knn_accuracy, cnn_accuracy, rnn_accuracy)

if __name__ == "__main__":
    main()
