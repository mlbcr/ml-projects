from myml.linear_models import LinearRegression, SGD, LogisticRegression
from myml.model_selection import train_test_split, cross_val_score
import numpy as np
from myml.metrics import accuracy_score

X = np.arange(10).reshape(-1, 1)
y = (np.arange(10) * 2).reshape(-1, 1)

model1 = LinearRegression()

print("R² médio:", cross_val_score(model1, X, y))


X = np.arange(10).reshape(-1, 1)
y = (X >= 5).astype(int)

model2 = LogisticRegression()
model2.fit(X, y)
y_pred = model2.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
