"""viz.py

This module provides the Visualization class for plotting the training history and the confusion matrix of a trained
machine learning model.

Example:
    from viz import Visualization
    viz = Visualization(trained_model)
    viz.plot_training_history(history_object)
    viz.plot_confusion_matrix(y_true, y_test)

Classes:
    Visualization: Class providing methods for visualizing the performance of a trained model.

This module requires `matplotlib` and `seaborn` to be installed within the Python environment you are using this module in.

Note:
    This module is meant to be used as a utility for machine learning model evaluation tasks.
"""


import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    """
    Class providing methods for visualizing the performance of a trained model.

    Attributes:
        model (object): A trained model.
    """
    def __init__(self, model):
        """
        Initializes the Visualization class with a trained model.

        Args:
            model (object): A trained model.
        """
        self.model = model

    def plot_training_history(self, history):
        """
        Plot the training history of the model.

        Args:
            history (History): A History object from keras. Its History.history attribute is a 
                               record of training loss values and metrics values at successive epochs.

        """
        fig, axs = plt.subplots(2)

        # create accuracy subplot
        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy evaluation")

        # create error subplot
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error evaluation")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot a confusion matrix using seaborn.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels by the model.
        """
        cf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
