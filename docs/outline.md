The provided requirements for the Python program are detailed and clear. Below are general Python class outlines that could be used as a starting point for your project.

## `KNN Class`

```python
class KNN:
    def __init__(self, k, distance_fn, vote_fn):
        pass  # Initialize attributes

    def train(self, X, y):
        pass  # Store training data and labels

    def predict(self, X):
        pass  # Calculate distances and assign labels
CNN Class
python
 
class CNN:
    def __init__(self, layers, activation_fn, loss_fn, optimizer):
        pass  # Initialize attributes

    def train(self, X, y):
        pass  # Initialize weights and biases

    def forward_propagation(self, X):
        pass  # Calculate outputs

    def compute_error(self, y_true, y_pred):
        pass  # Calculate error

    def back_propagation(self, error):
        pass  # Update weights and biases
RNN Class
python
 
class RNN:
    def __init__(self, layers, activation_fn, loss_fn, optimizer):
        pass  # Initialize attributes

    def train(self, X, y):
        pass  # Initialize weights and biases

    def forward_propagation(self, X):
        pass  # Calculate outputs

    def compute_error(self, y_true, y_pred):
        pass  # Calculate error

    def back_propagation(self, error):
        pass  # Update weights and biases
DataPreparation Class
python
 
class DataPreparation:
    def __init__(self, preprocessing_fn, split_ratio):
        pass  # Initialize attributes

    def preprocess(self, raw_data):
        pass  # Preprocess raw data

    def split(self, data, labels):
        pass  # Split data into training and testing sets
Unittests
You might have a unittest file per class, for example:

test_knn.py
test_cnn.py
test_rnn.py
test_data_preparation.py
Each would contain unittests applicable to its class.

Visualization Class
```python
 
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        pass  # Initialize if needed

    def plot_accuracy(self, train_acc, test_acc):
        pass  # Implement accuracy plotting

    # Implement other plotting functions as needed
```

## Main Configuration File

```python
from config import *  # Import configuration variables

# Import your classes
from knn import KNN
from cnn import CNN
from rnn import RNN
from data_preparation import DataPreparation
from visualization import Visualization

def main():
    # Load and preprocess data
    # Initialize models
    # Train models
    # Test models
    # Visualize results

if __name__ == "__main__":
    main()
```
