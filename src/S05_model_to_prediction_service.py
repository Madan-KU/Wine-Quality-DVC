import os
import json
import shutil
import logging

from modules.read_config import read_config
from modules.logger_configurator import configure_logger

    

def copy_best_model_to_prediction(report_metrics,saved_models_dir,serving_model_dir):
    """Copy best model to serving model directory """
    
    # Load metrics from metrics.json
    with open(os.path.join(report_metrics), 'r') as file:
        metrics_data = json.load(file)

    # Get r2 values
    r2_scores = {model: metrics_data[model]['metrics']['r2'] for model in metrics_data}

    # Find model with the highest r2 value
    best_model = max(r2_scores, key=r2_scores.get)
    best_model_file = os.path.join(saved_models_dir, best_model)
    serving_model= os.path.join(serving_model_dir,"model")

    # Copy best model to prediction_app directory
    shutil.copy(best_model_file, serving_model_dir)

    print(f"Copied '{best_model}' Model to '{serving_model}'")


def main():
    configure_logger()
    global config
    config = read_config('params.yaml')

    # Get paths from config
    report_metrics = config['reports']['metrics']
    saved_models_dir = config['saved_model_dir']
    serving_model_dir = config['prediction_app']['model']

    copy_best_model_to_prediction(report_metrics,saved_models_dir,serving_model_dir)

if __name__=="__main__":
    main()
