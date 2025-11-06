from myml.linear_models import LinearRegression, SGD
from myml.model_selection import train_test_split, cross_val_score
import numpy as np

X = np.arange(10).reshape(-1, 1)
y = (np.arange(10) * 2).reshape(-1, 1)

model1 = LinearRegression()

print("R² médio:", cross_val_score(model1, X, y))
