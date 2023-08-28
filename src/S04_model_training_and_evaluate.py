import os
import logging
import pickle
import yaml
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from modules.data_loader import read_data
from modules.logger_configurator import configure_logger

config= None
def read_yaml_config(file_path):
    """Read and return the configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


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
    
    #Saving Metrics to json file
    params_file_path =config['reports']['params']
    metrics_file_path= config['reports']['metrics']
    metrics_history_path=config['reports']['metrics_history']

    try:
        """Saving metrics history"""
        with open(metrics_history_path, "a") as file:
            metrics = {
                "timestamp": str(datetime.datetime.now()),
                "model": model_name,
                "metrics": {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                },
                # "features": features,
                "hyperparameters": model_params
            }
            file.write(json.dumps(metrics) +','+ '\n')
    except Exception as e:
        logging.error(f"Unable to save metrics history to {metrics_history_path}. Error: {e}")

    try:
        """First, read existing data if the file already exists"""
        if os.path.exists(metrics_file_path):
            with open(metrics_file_path, "r") as file:
                data = json.load(file)
        else:
            data = {}

        data[model_name] = {
            "metrics": {"rmse": rmse, "mae": mae,"r2": r2},
        }

        with open(metrics_file_path, "w") as file:
            json.dump(data, file, indent=4)

        
        if os.path.exists(params_file_path):
            with open(params_file_path, "r") as file:
                params_data = json.load(file)
        else:
            params_data = {}

        params_data[model_name] = {
            "hyperparameters": model_params
        }

        with open(params_file_path, "w") as file:
            json.dump(params_data, file, indent=4)

    except Exception as e:
        logging.error(f"Unable to save metrics or parameters. Error: {e}")
        
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
    config = read_yaml_config('params.yaml')
    saved_model_directory = config['saved_model_dir']

    df, filename = read_data(config['data']['transformed'])
    X_train, X_test, y_train, y_test = split_data(df, config)

    for model_name, model_config in config['model'].items():
        model_params = model_config.get('params', {})
        model_name, model = train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params)
        save_model(model_name, model, saved_model_directory)


if __name__ == "__main__":
    main()
