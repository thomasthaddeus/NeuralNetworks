"""knn_main.py

implementation of knn for now

_extended_summary_
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from knn import KNearestNeighbor
from numpy import floating, _16Bit, _32Bit, _64Bit

def main():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of KNearestNeighbors
    knn = KNearestNeighbors(k=3, problem_type='classification')

    # Train the model
    knn.train(X_train, y_train)

    # Make predictions on the test set
    predictions = knn.predict(X_test)

    # Evaluate the model
    accuracy: float | floating[_16Bit] | floating[_32Bit] | floating[_64Bit] = knn.evaluate_classification(X_test, y_test)

    print(f'Test accuracy: {accuracy}')

if __name__ == "__main__":
    main()
