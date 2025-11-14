import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    A simple implementation of sklearn.metrics accuracy_score.
    This is a simplified reimplementation for educational purposes :)

    Parameters:

    y_true: np.ndarray
        feature matrix
    y_pred: np.ndarray
        target vector

    Returns:

    accuracy_score: float
        the accuracy between y_pred and y_true values
    """
    return np.mean(y_pred == y_true)
