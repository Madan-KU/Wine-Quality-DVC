import yaml
import joblib
import prediction_app_config as app_config


def read_yaml_config(file_path):
    """Read and return the configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def predict(data):
    config = read_yaml_config('params.yaml')
    prediction_app_model=config['prediction_app_model']
    model=joblib.load(prediction_app_model)
    print("Model loaded")
    prediction=model.predict(data)
    print(prediction)
    return prediction


def api_response():
    pass


