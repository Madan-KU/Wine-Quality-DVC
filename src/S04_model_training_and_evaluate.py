import os
import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger
from modules.save_metrics import save_metrics


def get_model(model_name, model_params):
    """Return an instance of the model based on the provided name and parameters."""
    model_class = globals().get(model_name)
    if model_class:
        return model_class(**model_params)
    raise ValueError(f"Unknown model class: {model_name}")


def split_data(df, config):
    """Split the dataframe into train and test sets."""
    target = config['base']['target_col']
    X = df.drop(columns=target)
    y = df[target]
    return train_test_split(
        X, y,
        test_size=config['split_data']['test_size'],
        random_state=config['base']['random_state']
    )


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params):
    """Train the model and evaluate its performance."""
    model = get_model(model_name, model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"{model_name} - RMSE: {rmse:.2f} - MAE: {mae:.2f} - R2 Score: {r2:.2f}")

    save_metrics(model_name, model_params, rmse, mae, r2)

    return model_name, model



def save_model(model_name, model, saved_model_directory):
    """Save the trained model to a directory."""
    if not os.path.exists(saved_model_directory):
        os.makedirs(saved_model_directory)
    filepath = os.path.join(saved_model_directory, model_name)
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
        logging.info(f"'{model}' saved to '{filepath}'")


def main():
    """Main function to load data, train, and evaluate models."""
    configure_logger()
    global config
    config = read_config('params.yaml')
    saved_model_directory = config['saved_model_dir']

    df, filename = read_data(config['data']['transformed'])
    X_train, X_test, y_train, y_test = split_data(df, config)

    for model_name, model_config in config['model'].items():
        model_params = model_config.get('params', {})
        model_name, model = train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params)
        save_model(model_name, model, saved_model_directory)


if __name__ == "__main__":
    main()