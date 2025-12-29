import numpy as np

class GaussianAD:
    def __init__(self):
        self.mu = None
        self.var = None

    def calc_mean(self, X):
        # m, n = X.shape
        # mu = np.zeros(n)
        # for j in range(n):
        #     for i in range(m):
        #         mu[j] += X[i][j]
        #     mu[j] = mu[j] / m
        # return mu
        return X.mean(axis=0)

    def calc_var(self, X, mu):
        # m, n = X.shape
        # var = np.zeros(n)
        # for j in range(n):
        #     soma = 0
        #     for i in range(m):
        #         var[j] += (X[i][j] - mu[j]) ** 2
        #     var[j] = var[j] / m
        # return var
        return ((X - mu) ** 2).mean(axis=0)

    def fit(self, X):
        self.mu = self.calc_mean(X)
        self.var = self.calc_var(X, self.mu)

    def predict(self, X, threshold=0.5):
        p = np.ones(X.shape[0])
        for j in range(X.shape[1]):
            p *= 1 / np.sqrt(2 * np.pi * self.var[j]) * np.exp(-((X[:, j] - self.mu[j]) ** 2) / (2 * self.var[j]))

        return np.where(p < threshold, -1, 1)
