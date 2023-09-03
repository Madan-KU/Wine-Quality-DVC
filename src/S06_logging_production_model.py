
import pickle
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint

from modules.read_config import read_config
from modules.logger_configurator import configure_logger


class ModelLogger:

    def __init__(self, config):
        self.config = config
        self.mlflow_config = self.config["mlflow_configuration"]
        self.model_name = self.mlflow_config["registered_model_name"]
        self.remote_server_uri = self.mlflow_config["remote_server_uri"]

        mlflow.set_tracking_uri(self.remote_server_uri)
        print(f"MLflow Tracking URI set to: {self.remote_server_uri}")
        self.client = MlflowClient()

    def _get_lowest_mae_run_id(self):
        runs = mlflow.search_runs(experiment_ids=[13])
        print(f"All fetched runs: {runs}")
        
        runs["metrics.MAE"] = pd.to_numeric(runs["metrics.MAE"], errors='coerce')

        lowest = runs["metrics.MAE"].min()  # directly use the min() function
        best_run_id = runs[runs["metrics.MAE"] == lowest]["run_id"].iloc[0]  # use iloc[0] to get the first matching run ID
        print(f"Best run ID based on lowest MAE: {best_run_id}")
        return best_run_id

    def log_production_model(self):
        lowest_run_id = self._get_lowest_mae_run_id()
        logged_model = None

        model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        print(f"Found {len(model_versions)} model versions for model name '{self.model_name}'")

        for mv in model_versions:
            mv = dict(mv)
            print(f"Checking model version from run: {mv['run_id']}")

            if mv["run_id"] == lowest_run_id:
                current_version = mv["version"]
                logged_model = mv["source"]
                print(f"Model version found for best run: {mv['run_id']}")
                pprint(mv, indent=4)
                print(f"Transitioning model version {current_version} to Production...")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=current_version,
                    stage="Production"
                )
                print(f"Model version {current_version} transitioned to Production.")
            else:
                print(f"Model version not matching best run. Skipping: {mv['run_id']}")
                current_version = mv["version"]
                print(f"Transitioning model version {current_version} to Staging...")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=current_version,
                    stage="Staging"
                )
                print(f"Model version {current_version} transitioned to Staging.")

        if logged_model:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            model_path = os.path.join(self.config["production_model"], "model.pkl")
            with open(model_path, 'wb') as file:
                pickle.dump(loaded_model, file)
            print(f"Model saved to: {model_path}")
        else:
            print("No model was set to production.")



if __name__ == '__main__':
    configure_logger()
    config = read_config('params.yaml')
    model_logger = ModelLogger(config)
    model_logger.log_production_model()
