from sklearn.svm import SVR

def train_support_vector_regression_model(X_train, y_train, model_params):
    model = SVR(**model_params)
    model.fit(X_train, y_train)
    return model
