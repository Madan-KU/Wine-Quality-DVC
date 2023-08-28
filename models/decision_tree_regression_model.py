from sklearn.tree import DecisionTreeRegressor

def train_decision_tree_regression_model(X_train, y_train, model_params):
    model = DecisionTreeRegressor(**model_params)
    model.fit(X_train, y_train)
    return model
