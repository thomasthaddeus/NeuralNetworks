"""test_knn.py

imports the k nearest neighbors module

_extended_summary_
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification
from knn import KNearestNeighbors

class TestKNearestNeighbors(unittest.TestCase):
    def setUp(self):
        self.knn = KNearestNeighbors(3, problem_type='classification')
        self.X, self.y = make_classification(n_samples=50, n_features=4, random_state=42)

    def test_train(self):
        self.knn.train(self.X, self.y)
        self.assertTrue(self.knn.is_trained)

    def test_predict(self):
        self.knn.train(self.X, self.y)
        predictions = self.knn.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_evaluate_classification(self):
        self.knn.train(self.X, self.y)
        accuracy = self.knn.evaluate_classification(self.X, self.y)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()
