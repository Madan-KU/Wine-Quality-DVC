import os
import json
import yaml
import pickle
import logging

# import warnings
# warnings.simplefilter(action='ignore', category=Warning)

def read_yaml_config(file_path):
    """Read and return the configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_pickle(pickle_path):
    """Loads pickled data into memory for predictions."""
    with open(pickle_path, 'rb') as pickle_file:
        loaded_pickle_file=pickle.load(pickle_file)
        return loaded_pickle_file
    

def read_schema(input_schema_path):
    with open(input_schema_path,'r') as file:
        schema=json.load(file)
        return schema
    
def validate_input(schema, data):
    return True




def predict(data):
    config = read_yaml_config('params.yaml')
    prediction_model_path=os.path.join(config['prediction_app']['model'],"model.pkl")
    scaler_path=os.path.join(config['prediction_app']['scaler'],"X_scaler.pkl")
    input_schema_path = config['schema']['input']
    

    model=load_pickle(prediction_model_path)
    scaler=load_pickle(scaler_path)
    schema=read_schema(input_schema_path)
    validate_input(schema,data)
    
    preprocessed_data = scaler.transform(data) 
    prediction = model.predict(preprocessed_data)
    print(prediction)
    return prediction

# data = {
#     "fixed_acidity": 8.5,
#     "volatile_acidity": 0.28,
#     "citric_acid": 0.56,
#     "residual_sugar": 1.8,
#     "chlorides": 0.092,
#     "free_sulfur_dioxide": 35.0,
#     "total_sulfur_dioxide": 103.0,
#     "density": 0.9969,
#     "pH": 3.3,
#     "sulphates": 0.75,
#     "quality": 7
#     }


# try:
#     config = read_yaml_config('params.yaml')
#     input_schema_path = config['schema']['input']
#     schema=read_schema(input_schema_path)
#     # print(schema)
#     validate_input(data,schema)

#     prediction = predict([data])
#     print(prediction)
# except ValueError as e:
#     print(f"Validation Error: {e}")



