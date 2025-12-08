from statistics import harmonic_mean

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


def precision_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    A simple implementation of sklearn.metrics precision score.
    This is a simplified reimplementation for educational purposes :)

    Parameters:

    y_true: np.ndarray
        feature matrix
    y_pred: np.ndarray
        target vector

    Returns:

    precision_score: float
        the precision between y_pred and y_true values
    """
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    fp = np.sum((y_pred == y_true) & (y_true == 0))

    return tp / (tp + fp)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    A simple implementation of sklearn.metrics recall score.
    This is a simplified reimplementation for educational purposes :)

    Parameters:

    y_true: np.ndarray
        feature matrix
    y_pred: np.ndarray
        target vector

    Returns:

    recall_score: float
        the recall between y_pred and y_true values
    """
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    fn = np.sum((y_pred != y_true) & (y_true == 1))

    return tp / (tp + fn)



def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
        A simple implementation of sklearn.metrics f1_score score.
        This is a simplified reimplementation for educational purposes :)

        Parameters:

        y_true: np.ndarray
            feature matrix
        y_pred: np.ndarray
            target vector

        Returns:

        f1_score_score: float
            the f1 score between y_pred and y_true values
        """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r)