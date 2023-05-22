"""config.py

_summary_

_extended_summary_

Returns:
    _type_: _description_
"""
from sklearn.model_selection import train_test_split
from typing import Any

class DataPreparation:
    """
    Class for data preparation tasks.

    Args:
        preprocessing_fn (function): The preprocessing function to be applied to the raw data.
        split_ratio (float): The ratio for splitting the data into training and testing sets.

    Attributes:
        preprocessing_fn (function): The preprocessing function to be applied to the raw data.
        split_ratio (float): The ratio for splitting the data into training and testing sets.
    """

    def __init__(self, preprocessing_fn, split_ratio) -> None:
        """
        Initializes a DataPreparation instance.

        Args:
            preprocessing_fn (function): The preprocessing function to be applied to the raw data.
            split_ratio (float): The ratio for splitting the data into training and testing sets.
        """
        self.preprocessing_fn: Any = preprocessing_fn
        self.split_ratio: Any = split_ratio

    def preprocess(self, raw_data):
        """
        Preprocesses the raw data using the specified preprocessing function.

        Args:
            raw_data: The raw data to be preprocessed.

        Returns:
            The preprocessed data.
        """
        return self.preprocessing_fn(raw_data)

    def split(self, data, labels) -> tuple:
        """
        Splits the data and labels into training and testing sets.

        Args:
            data: The data to be split.
            labels: The corresponding labels.

        Returns:
            A tuple containing the training and testing data and labels: (x_train, x_test, y_train, y_test).
        """
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.split_ratio, random_state=42)
        return x_train, x_test, y_train, y_test
