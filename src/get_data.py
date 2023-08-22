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

csvfile= ''

def read_data_from_directory(directory_path):

    if not os.path.exists(directory_path):
        logging.warning(f"Directory {directory_path} does not exist.")
        return None

    for csvfile in os.listdir(directory_path):
        if csvfile.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(directory_path, csvfile))
                logging.info(f"Successfully read {csvfile}")
                return df,csvfile
            except Exception as e:
                logging.error(f"Error reading {csvfile}: {e}")
    
    logging.warning(f"No CSV file found in {directory_path}.")
    print(csvfile)
    return None

def main():
    df,csvfile = read_data_from_directory(config['data']['remote'])

    if df is not None:
        print(df)
        
    output_path = os.path.join(config['data']['raw'], csvfile)
    df.to_csv(output_path)

    logging.info(f"'{csvfile}' loaded to '{config['data']['raw']}'")

if __name__ == "__main__":
    main()

