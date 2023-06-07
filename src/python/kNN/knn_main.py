"""knn_main.py

This script demonstrates the implementation of the k-Nearest Neighbors (k-NN) algorithm
on the iris dataset using a custom KNearestNeighbors class.

The script includes the main function which loads the iris dataset, splits it into a training
and test set, creates an instance of the KNearestNeighbors class, trains the model, makes
predictions on the test set, and evaluates the model's performance based on accuracy.
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from knn import KNearestNeighbors


def knn_main():
    """
    The main function for running the k-Nearest Neighbors model on the iris dataset.

    This function loads the iris dataset, splits it into training and test sets,
    trains a k-Nearest Neighbors model on the training set, makes predictions
    on the test set, and evaluates the model's performance by calculating the
    accuracy of its predictions.
    """
    # [pylint: disable:invalid-name]
    # Load the iris dataset
    iris = load_iris()

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Create an instance of KNearestNeighbors
    knn = KNearestNeighbors(k=3, problem_type='classification')

    # Fit the model to the training data
    knn.train(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate the model
    accuracy2 = knn.evaluate_classification(X_test, y_test)

    print(f"Test accuracy: {accuracy}")
    print(f"Test accuracy2: {accuracy2}")

if __name__ == "__main__":
    knn_main()
