from myml.linear_models import LinearRegression, SGD
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

model1 = LinearRegression()
model2 = SGD()
model1.fit(X, y)
model2.fit(X, y)
y_pred = model1.predict(X)
print(model1.score(y_pred, y))
y_pred = model2.predict(X)
print(model2.score(y_pred, y))