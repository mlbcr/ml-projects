import numpy as np
from .base import BaseModel
# -----------------------------------------------------------------------------------------
# Linear Regression with Normal Equation
class EqLinearRegression(BaseModel):
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


# -----------------------------------------------------------------------------------------
# Linear Regression with Batch Gradient Descent
class LinearRegression(BaseModel):
    def __init__(self, eta=0.01, n_iterations=1000):
        self.theta = None
        self.n_iterations = n_iterations
        self.eta = eta

    def fit(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        n_features = X_b.shape[1]
        self.theta = np.random.randn(n_features, 1)

        for i in range(self.n_iterations):
            gradients = (2/m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta = self.theta - self.eta * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
# -----------------------------------------------------------------------------------------
# Linear Regression with Stochastic Gradient Descent
class SGD(BaseModel):
    def __init__(self, eta=0.1, n_epochs=500):
        self.theta = None
        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        n_features = X_b.shape[1]
        self.theta = np.random.randn(n_features, 1)
        t0, t1 = 1, 1000

        def learning_schedule(t):
            return t0 / (t + t1)

        t = 0
        for epoch in range(self.n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X_b[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                eta_t = learning_schedule(t)
                self.theta = self.theta - eta_t * gradients
                t += 1

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

# -----------------------------------------------------------------------------------------
# Logistic Regression 
class LogisticRegression(BaseModel):
    def __init__(self, eta=0.01, n_iterations=1000):
        self.theta = None
        self.n_iterations = n_iterations
        self.eta = eta

    def fit(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        y = y.reshape(-1, 1)

        self.theta = np.random.randn(X_b.shape[1], 1)

        for i in range(self.n_iterations):
            z = X_b.dot(self.theta)
            probs = 1 / (1 + np.exp(-z))
            gradients = (1 / m) * X_b.T.dot(probs - y)
            self.theta -= self.eta * gradients

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return 1 / (1 + np.exp(-X_b.dot(self.theta)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.50).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
