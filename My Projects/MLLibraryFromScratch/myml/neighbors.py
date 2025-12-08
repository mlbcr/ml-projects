import numpy as np

class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def _euclidean_distance(self, n1, n2):
        return np.sqrt(np.sum((n1 - n2) ** 2))