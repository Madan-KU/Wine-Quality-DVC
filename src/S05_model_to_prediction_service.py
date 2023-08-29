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
    best_model_path = os.path.join(saved_models_dir, best_model+'.pkl')
    serving_model_path= os.path.join(serving_model_dir,'model.pkl')

    if not os.path.exists(serving_model_dir):
        os.makedirs(serving_model_dir,exist_ok=True)

    # Copy best model to prediction_app directory
    shutil.copy(best_model_path, serving_model_path)
    logging.info(f"Copied '{best_model}' model to '{serving_model_path}'")


def copy_scaler_to_prediction(scaler_dir,serving_scaler_dir):
    """Copy scaler files into serving folder."""

    if not os.path.exists(serving_scaler_dir):
        os.makedirs(serving_scaler_dir,exist_ok=True)
    
    source_scaler_file_path = os.path.join(scaler_dir, "scaler.pkl")
    destination_scaler_file_path = os.path.join(serving_scaler_dir, "scaler.pkl")

    # Copy the scaler file to the serving directory
    shutil.copy(source_scaler_file_path, destination_scaler_file_path)
    
    # Log the action
    logging.info(f"Scaler copied from '{source_scaler_file_path}' to '{destination_scaler_file_path}'.")



def main():
    configure_logger()
    global config
    config = read_config('params.yaml')

    # Get paths from config
    report_metrics = config['reports']['metrics']
    saved_models_dir = config['saved_model_dir']
    serving_model_dir = config['prediction_app']['model']
    scaler_dir=config['scaler_dir']
    serving_scaler_dir = config['prediction_app']['scaler']

    copy_best_model_to_prediction(report_metrics,saved_models_dir,serving_model_dir)
    copy_scaler_to_prediction(scaler_dir,serving_scaler_dir)

if __name__=="__main__":
    main()
