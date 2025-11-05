from myml.linear_models import LinearRegression, SGD
from myml.model_selection import train_test_split
import numpy as np
X = np.arange(10)
y = np.arange(10) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train:", X_train)
print("X_test:", X_test)

model1 = LinearRegression()
model2 = SGD(n_epochs=50)
model1.fit(X, y)
model2.fit(X, y)

y_pred = model1.predict(X)
print(model1.score(y_pred, y))

y_pred = model2.predict(X)
print(model2.score(y_pred, y, "mse"))