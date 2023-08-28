from sklearn.linear_model import Ridge

def train_ridge_regression_model(X_train, y_train, model_params):
    model = Ridge(**model_params)
    model.fit(X_train, y_train)
    return model
