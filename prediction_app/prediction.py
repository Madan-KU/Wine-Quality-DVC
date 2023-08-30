import os
import yaml
import pickle
import prediction_app_config as app_config


def read_yaml_config(file_path):
    """Read and return the configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_pickle(pickle_path):
    """Loads pickled data into memory for predictions."""
    with open(pickle_path, 'rb') as pickle_file:
        loaded_pickle_file=pickle.load(pickle_file)
        return loaded_pickle_file


def predict(data):
    config = read_yaml_config('params.yaml')
    prediction_model_path=os.path.join(config['prediction_app']['model'],"model.pkl")
    scaler_path=os.path.join(config['prediction_app']['scaler'],"X_scaler.pkl")
    
    model=load_pickle(prediction_model_path)
    scaler=load_pickle(scaler_path)
    
    preprocessed_data = scaler.transform(data) 
    prediction = model.predict(preprocessed_data)
    print(prediction)
    return prediction





