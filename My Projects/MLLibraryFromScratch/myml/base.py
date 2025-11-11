import numpy as np

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError("fit() method needs to be implemented.")

    def predict(self, X):
        raise NotImplementedError("predict() method needs to be implemented.")

    def score(self, X, y, type="r2"):
        y_pred = self.predict(X)
        if type == "r2":
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            return r2
        elif type == "mse":
            mse = np.mean((y - y_pred) ** 2)
            return mse
        return None
