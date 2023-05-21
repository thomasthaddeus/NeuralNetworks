# Neural Networks

- [Neural Networks](#neural-networks)
  - [General Requirements](#general-requirements)
  - [Detailed Requirements](#detailed-requirements)
    - [`k-NN Class`](#k-nn-class)
    - [`CNN Class`](#cnn-class)
    - [`RNN Class`](#rnn-class)
    - [Training Data Class](#training-data-class)
    - [Unittests](#unittests)
    - [Visualization Class](#visualization-class)
  - [Python Program Requirements](#python-program-requirements)
  - [Python-based program implementing](#python-based-program-implementing)
    - [Focus on](#focus-on)
    - [Algorithms Implementation](#algorithms-implementation)
    - [Data Preparation](#data-preparation)
    - [Main Configuration File](#main-configuration-file)
    - [Unit Tests](#unit-tests)
    - [Visualization](#visualization)
    - [Documentation](#documentation)
    - [Folder Structure](#folder-structure)

## General Requirements

1. The program should be written in Python.
2. The program should be composed primarily of the implementation of three algorithms:
   - k-Nearest Neighbors (k-NN)
   - Convolutional Neural Network (CNN)
   - Recurrent Neural Network (RNN).
3. Each algorithm should be implemented as its own class or a series of classes with a common superclass.
4. The implementation of these algorithms should be controlled by reprogrammable functions inside the classes.
5. Training data should be handled in a separate class before production.
6. There should be a main file that runs the configuration for this program and determines the input for variables.
7. Unittests should be written for error checking of each file.
8. A suitable class should be created for visualizations of the program's output.

## Detailed Requirements

### `k-NN Class`

1. A function should be implemented that takes in training data and labels, storing them for future use.
2. A function should be implemented that takes in test data and calculates the distance from each point in the test data to each point in the stored training data.
3. A function should be implemented that assigns a label to each test data point based on the labels of the 'k' nearest training data points.
4. Reprogrammable functions should be used to allow flexibility in distance calculation method and the method of assigning labels based on the 'k' nearest neighbors.

### `CNN Class`

1. A class or series of classes should be implemented to define the layers of the CNN, including convolutional layers, pooling layers, and fully connected layers.
2. A function should be implemented that takes in training data and labels, initializing the weights and biases of the network.
3. A function should be implemented that performs forward propagation through the network for a given input.
4. A function should be implemented that calculates the error of the network's output compared to the expected output.
5. A function should be implemented that performs backpropagation through the network, updating the weights and biases based on the error.
6. Reprogrammable functions should be used to allow flexibility in activation function, loss function, and optimization method.

### `RNN Class`

1. A class or series of classes should be implemented to define the layers of the RNN, including recurrent layers and fully connected layers.
2. A function should be implemented that takes in training data and labels, initializing the weights and biases of the network.
3. A function should be implemented that performs forward propagation through the network for a given input.
4. A function should be implemented that calculates the error of the network's output compared to the expected output.
5. A function should be implemented that performs backpropagation through time in the network, updating the weights and biases based on the error.
6. Reprogrammable functions should be used to allow flexibility in activation function, loss function, and optimization method.

### Training Data Class

1. A function should be implemented that takes in raw data and preprocesses it into a suitable format for input into the algorithms.
2. A function should be implemented that splits the data into training and testing sets.
3. Reprogrammable functions should be used to allow flexibility in preprocessing methods and split ratios.

### Unittests

1. Unittests should be written for each class to ensure the correctness of the algorithm implementations.
2. Unittests should be written for the training data class to ensure the correctness of the preprocessing and splitting functions.

### Visualization Class

1. A class should be created for generating visualizations of the program's output.
2. Functions should be implemented to generate different types of visualizations, such as plots of the training and testing accuracy over time

## Python Program Requirements

## Python-based program implementing

- k-Nearest Neighbors (k-NN),
- Convolutional Neural Network (CNN),
- Recurrent Neural Network (RNN)

### Focus on

- reprogrammable functions within classes
- training data
- a main configuration file
- unit tests
- visualization

### Algorithms Implementation

1. Create separate classes for each algorithm: KNN, CNN, and RNN.
2. These classes should allow for re-programmability of the algorithm's function(s).
3. Each class should have methods for training the model and predicting the output.

### Data Preparation

1. Create a `DataPreparation` class for handling the following tasks related to training data:
   - loading
   - cleaning
   - preprocessing

### Main Configuration File

1. Create a main.py file that imports the classes, imports the program configuration from config.py, and have control over the execution of the program.
1. A separate config.py file will determine the input for variables .

### Unit Tests

1. For each class (KNN, CNN, RNN, DataPreparation), write unit tests in separate Python files.
1. The tests should aim to cover all functions within each class and check for errors.

### Visualization

Create a Visualization class for creating visualizations of the program's output.

1. This class should make use of one or more visualization libraries such as Matplotlib, Seaborn, or Plotly depending on the specific requirements of the visualizations.
   - Matplotlib is very versatile and can plot any kind of graph, but complex plots might require more code than other libraries​​.
   - Seaborn provides a higher-level interface for similar plots as Matplotlib, resulting in less code and a nicer design for common plots such as bar plot, box plot, histograms, etc​.
   - Plotly is excellent for creating interactive and publication-quality graphs. It can create similar charts as Matplotlib and Seaborn with the added benefit of interactivity​.

1. This class should provide methods for generating specific visualizations based on the outputs of the algorithms (k-NN, CNN, RNN).

### Documentation

1. Ensure that all classes, functions, and methods are well-documented.
1. The docstrings should be in the style of Google documentation.
1. This includes writing descriptive docstrings for each class and function, and commenting the code where necessary.

### Folder Structure

```plaintext
project_root/
│
├── main.py                           # Main entry point to the program
│
├── config.py                         # Configuration file
│
├── data/
│   ├── train/                        # Training data files
│   └── test/                         # Test data files
│
├── python/                           # Algorithms implementation
│   ├── base_algorithm.py             # Base algorithm class
│   ├── k_nn.py                       # k-Nearest Neighbors implementation
│   ├── cnn.py                        # Convolutional Neural Network implementation
│   └── rnn.py                        # Recurrent Neural Network implementation
│
├── visual/                           # Data visualization
│   ├── base_viz.py                   # Base visualization class
│   ├── hist_viz.py                   # Histogram visualization class
│   ├── sctr_plot_viz.py              # Scatter plot visualization class
│   └── heatmap_viz.py                # Heatmap visualization class
│
├── tests/                            # Unit tests
│   ├── test_k_nn.py                  # Tests for k-NN
│   ├── test_cnn.py                   # Tests for CNN
│   ├── test_rnn.py                   # Tests for RNN
│   └── test_viz.py                   # Tests for visualization
│
└── README.md                         # Project documentation
```

In this structure, base_algorithm.py contains a superclass from which the specific algorithm classes (k_nn.py, cnn.py, rnn.py) inherit.

The visualization folder contains several Python files, each corresponding to a different kind of visualization.
These files inherit from a base visualization class defined in base_viz.py.
Exact breakdown of visualization types (e.g., histograms, scatter plots, heatmaps) will depend on specific python files.

The test_viz.py file contains unit tests for the visualization classes.
