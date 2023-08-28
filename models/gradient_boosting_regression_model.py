from sklearn.ensemble import GradientBoostingRegressor

def train_gradient_boosting_regression_model(X_train, y_train, model_params):
    model = GradientBoostingRegressor(**model_params)
    model.fit(X_train, y_train)
    return model
