import os
import json
import yaml
import logging
from modules.data_loader import read_data
from modules.logger_configurator import configure_logger

configure_logger()

# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = read_yaml_config('params.yaml')
    schema_path = config['schema']['input']
    
    try:
        df, file = read_data(config['data']['cleansed'])
        
        df_describe = df.describe()
        df_describe_json = df_describe.loc[["min","max"]].to_json()

        with open(schema_path, "w+") as file:
            file.write(df_describe_json)
            logging.info(f"Schema written to '{schema_path}'")
    except Exception as e:
        logging.error(f"Unable to write input schema. Error: {e}")

if __name__ == "__main__":
    main()