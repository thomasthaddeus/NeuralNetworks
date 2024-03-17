# Neural Networks
Python program that implements a simple convolutional neural network (CNN) using the Keras library

<!-- BADGIE TIME -->
<!-- END BADGIE TIME -->

## REQUIREMENTS

1. [Link to Requirements](./docs/requirements.md)
2. [Data tree](./docs/tree.html)
3. [Overview](./src/pipeline/overview.ipynb)

## Data for testing

To be able to test these algorithms I used a dataset I found on Kaggle
The dataset is Located [Here]
 or you can download it using the following command:

```bash
kaggle datasets download -d puneet6060/intel-image-classification
```

## CNN

Ex. of a Python program that implements a simple convolutional neural network (CNN) using the Keras library

In this example, we're using the `Keras` library to define and train the CNN. \
Load the MNIST dataset using the `mnist.load_data()` function from Keras.
Then preprocess the data by normalizing the pixel values between **0 and 1**` and adding a channel dimension for grayscale images.

1. The model consists of several layers:
    - Two convolutional layers with max pooling
    - Followed by a flattening layer
    - Then two fully connected (dense) layers
    - The final layer uses the softmax activation function to output probabilities for each class
2. To use this code, you would need to load and preprocess your dataset accordingly.
3. Replace the comments in the code with the necessary code for loading and preprocessing your data.
4. Finally, you can train the model using the fit method and evaluate its performance using the evaluate method.

```python
# Load the dataset (assuming MNIST)
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# Normalize pixel values between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Add channel dimension for grayscale images
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
```

Next, we convert the class labels to one-hot encoded vectors using the to_categorical function from Keras. This step is necessary when dealing with multi-class classification problems.
Finally, we train the model using the preprocessed training data and labels. The fit method is used to train the model for a specified number of epochs and a specified batch size. We also provide the test data and labels as the validation data to monitor the model's performance during training.
After training, we evaluate the model on the test data using the evaluate method, and print the test loss and accuracy.
Remember to ensure that you have the required dependencies installed and import the necessary libraries before running the code.

```python
# Load the dataset (assuming MNIST)
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# Normalize pixel values between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Add channel dimension for grayscale images
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
```

Reshaped the input data: The shape of the input data is modified to include the channel dimension explicitly. This is achieved by using the reshape method on the X_train and X_test arrays.

Modified the architecture: The architecture of the CNN model has been slightly modified. We added a second convolutional layer and changed the size of the fully connected layer to 128 units.

Added verbosity during training: We set the verbose argument to 1 during the training process. This allows us to see the progress and logs during the training.

Other than these changes, the rest of the code remains the same. The dataset is loaded, preprocessed, and the model is compiled, trained, and evaluated in a similar manner.

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

class CNNModel:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')

    def predict(self, X):
        return self.model.predict(X)

# Example usage
cnn = CNNModel()
cnn.build_model(input_shape=(28, 28, 1), num_classes=10)
X_train, y_train, X_test, y_test = cnn.load_data()
cnn.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=128)
cnn.evaluate(X_test, y_test)
```

In this example, we have encapsulated the CNN model and related operations within the CNNModel class. The class has methods for building the model, loading the data, training, evaluating, and making predictions.

You can create an instance of the CNNModel class and call its methods to perform the desired operations. The build_model method allows you to specify the input shape and the number of classes. The load_data method loads and preprocesses the MNIST dataset. The train method trains the model with the provided data, while the evaluate method evaluates the model's performance on the test data. Finally, the predict method can be used to make predictions on new data.

This class provides a convenient way to reuse the CNN model and its functions by simply creating an instance of the class and calling the appropriate methods with the desired values.

[Here]:  <https://www.kaggle.com/datasets/puneet6060/intel-image-classification> "Intel Image Classification"
