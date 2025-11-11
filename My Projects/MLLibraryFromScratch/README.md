# MLLibrary
I'm currently developing a Machine Learning library based on scikit-learn *for educational purposes*, so that I can undestand each implementation of scikit-learn functions and models. 


## Implemented so far
### myml.base
- BaseModel(): I created a base model to implement the other models in a similar structure

### myml.linear_models
- LinearRegression(): I implemented using both Gradient Descent and Normal Equation to train math.
- LogisticRegression()


### myml.metrics
- accuracy_score(y_true, y)

### myml.model_selection
- train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False, stratify=None)
- cross_val_score(model, X, y, cv=5)

## Use Case
Example usage will be added soon, including:
- Loading a dataset
- Training a model
- Evaluating it
