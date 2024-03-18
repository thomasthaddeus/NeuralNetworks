"""test_cnn_model.py

This module contains unit tests for the CNNModel class.
"""

import unittest
import numpy as np
from python.models.cnn import CNNModel


class TestCNNModel(unittest.TestCase):
    def setUp(self):
        self.cnn = CNNModel()
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.epochs = 1
        self.batch_size = 128

    def test_build_model(self):
        # Test whether the model can be built successfully
        try:
            self.cnn.build_model(self.input_shape, self.num_classes)
        except Exception as e:
            self.fail(f"test_build_model failed with {e}")

        self.assertIsNotNone(self.cnn.model)

    def test_load_data(self):
        # Test data loading
        try:
            x_train, y_train, x_test, y_test = self.cnn.load_data()
        except Exception as e:
            self.fail(f"test_load_data failed with {e}")

        # Check the shapes of the loaded data
        self.assertEqual(x_train.shape[1:], self.input_shape)
        self.assertEqual(y_train.shape[1], self.num_classes)

    def test_train(self):
        # Test model training
        x_train, y_train, x_test, y_test = self.cnn.load_data()
        self.cnn.build_model(self.input_shape, self.num_classes)
        try:
            self.cnn.train(
                x_train, y_train, x_test, y_test, self.epochs, self.batch_size
            )
        except Exception as e:
            self.fail(f"test_train failed with {e}")

    def test_evaluate(self):
        # Test model evaluation
        x_train, y_train, x_test, y_test = self.cnn.load_data()
        self.cnn.build_model(self.input_shape, self.num_classes)
        self.cnn.train(
            x_train, y_train, x_test, y_test, self.epochs, self.batch_size
        )

        try:
            self.cnn.evaluate(x_test, y_test)
        except Exception as e:
            self.fail(f"test_evaluate failed with {e}")

    def test_predict(self):
        # Test model prediction
        x_train, y_train, x_test, y_test = self.cnn.load_data()
        self.cnn.build_model(self.input_shape, self.num_classes)
        self.cnn.train(
            x_train, y_train, x_test, y_test, self.epochs, self.batch_size
        )

        # Generate a random example for prediction
        random_example = np.random.rand(1, *self.input_shape)
        try:
            prediction = self.cnn.predict(random_example)
        except Exception as e:
            self.fail(f"test_predict failed with {e}")

        # Check if the prediction has the right shape
        self.assertEqual(prediction.shape, (1, self.num_classes))


if __name__ == "__main__":
    unittest.main()
