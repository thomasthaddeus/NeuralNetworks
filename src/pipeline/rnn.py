    """
     _summary_

    _extended_summary_
    """

import input_output

# Create a CNNToRNN model
model = CNNToRNN(image_shape=(64, 64, 3), num_classes=10)

# Create a RNNToKNN model
classifier = RNNToKNN(n_neighbors=3, model=model)

# Train the classifier on some data
X_train = ...  # The training images
y_train = ...  # The training labels
classifier.fit(X_train, y_train)

# Make predictions on some new data
X_test = ...  # The test images
predictions = classifier.predict(X_test)
