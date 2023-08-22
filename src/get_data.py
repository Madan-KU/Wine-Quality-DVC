#read params,
#process data
##return data 

import os
import logging
import argparse
import pandas as pd
import yaml


# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
config = read_yaml_config('parameters.yaml')
# print(f"config.yaml:{config}\n")


# Configuring logging using YAML values
logging.basicConfig(level=config['logging']['level'],
                    format=config['logging']['format'],
                    handlers=[logging.FileHandler(config['logging']['log_file']), logging.StreamHandler()])

filename= ''

def read_data_from_directory(directory_path):

    if not os.path.exists(directory_path):
        logging.warning(f"Directory {directory_path} does not exist.")
        return None

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(directory_path, filename))
                logging.info(f"Successfully read {filename}")
                return df,filename
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    
    logging.warning(f"No CSV file found in {directory_path}.")
    return None

def main():
    df,filename = read_data_from_directory(config['data']['remote'])
    filename='raw_'+ filename

    if df is not None:
        print(df)
        
    output_path = os.path.join(config['data']['raw'], filename)
    df.to_csv(output_path, index=False)

    logging.info(f"'{filename}' loaded to '{config['data']['raw']}'")

if __name__ == "__main__":
    main()

