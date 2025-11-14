import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float =0.2, random_state: int = None,
                     shuffle: bool = False, stratify: np.ndarray = None):
    """
    A simple implementation of sklearn.model_selection train_test_split.
    This is a simplified reimplementation for educational purposes :)

    Parameters:

    X: np.ndarray
        feature matrix
    y: np.ndarray
        target vector
    test_size: float
        dataset percentage for test
    random_state: int (optional, the standard is 42)
        random for reproducibility
    shuffle: boolean
        shuffles dataset before splitting
    stratify (not implemented yet): np.ndarray, default None
        Not implemented yet, ensures that the class proportions are the same as in the original dataset
        (only for classification problems).

    Returns:

    X_train, X_test, y_train, y_test: tuple
        splitted dataset
    """

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_count = int(n_samples * test_size)
    train_count = n_samples - test_count

    X_train, X_test = X[indices[:train_count]], X[indices[train_count:]]
    y_train, y_test = y[indices[:train_count]], y[indices[train_count:]]

    return X_train, X_test, y_train, y_test


def cross_val_score(model, X: np.ndarray, y: np.ndarray, cv=5):
    """
    A simple implementation of sklearn.model_selection cross_val_score.
    This is a simplified reimplementation for educational purposes :)

    Parameters:

    model: object
        any model that was implemented in the modules that have fit(X, y) and score(X, y) methods
    X: np.ndarray
        feature matrix
    y: np.ndarray
        target vector
    cv: int
        number of cross-validation folds

    Returns:

    mean_scores: float
        the mean of scores obtained on each fold
    """
    n = X.shape[0]
    fold_size = n // cv
    scores = []

    for i in range(cv):
        start = i * fold_size
        end = (i + 1) * fold_size if i != cv - 1 else n

        X_test = X[start:end]
        y_test = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        print("X_train:\n", X_train)
        print("X_test:\n", X_test)
        print('--------------------------')

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

    return np.mean(scores)