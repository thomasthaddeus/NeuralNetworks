"""data_preparation.py

Summary:
    This script defines the DataPreparation class which assists in the
    preprocessing and splitting of data for machine learning models.

Extended Summary:
    The DataPreparation class takes a preprocessing function and a split ratio
    as inputs during initialization. It has methods to preprocess the raw data
    using the provided function and to split the processed data into training
    and testing sets according to the provided ratio.

Returns:
    None. The script defines a class but does not run any operations by itself.
"""

from sklearn.model_selection import train_test_split


class DataPreparation:
    """
    Class for data preparation tasks.

    Args:
        preprocessing_fn (function):
            The preprocessing function to be applied to the raw data.
        split_ratio (float):
            The ratio for splitting the data into training and testing sets.

    Attributes:
        preprocessing_fn (function):
            The preprocessing function to be applied to the raw data.
        split_ratio (float):
            The ratio for splitting the data into training and testing sets.
    """

    def __init__(self, preprocessing_fn: callable, split_ratio):
        """
        Initializes a DataPreparation instance.

        Args:
            preprocessing_fn (function):
                The preprocessing function to be applied to the raw data.
            split_ratio (float):
                The ratio for splitting the data into training and testing sets.
        """
        self.preprocessing_fn = preprocessing_fn
        self.split_ratio = split_ratio

    def preprocess(self, raw_data):
        """
        Preprocesses the raw data using the specified preprocessing function.

        Args:
            raw_data: The raw data to be preprocessed.

        Returns:
            The preprocessed data.
        """
        return self.preprocessing_fn(raw_data)

    def split(self, data, labels):
        """
        Splits the data and labels into training and testing sets.

        Args:
            data: The data to be split.
            labels: The corresponding labels.

        Returns:
            A tuple containing the training and testing data and labels:
                (x_train, x_test, y_train, y_test).
        """
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=self.split_ratio, random_state=42
        )
        return x_train, x_test, y_train, y_test
