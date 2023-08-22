import os
import logging
# import argparse
import pandas as pd
import yaml

# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
config = read_yaml_config('parameters.yaml')

logging.basicConfig(level=config['logging']['level'],
                    format=config['logging']['format'],
                    handlers=[logging.FileHandler(config['logging']['log_file']), logging.StreamHandler()])

def read_data():
    raw_file_path=config['data']['raw']

    if not os.path.exists(raw_file_path):
        logging.warning(f"Directory {raw_file_path} does not exist.")
        return None
    
    for file in os.listdir(raw_file_path):

        print(f'1{raw_file_path}')
        
        if file.endswith('.csv'):
            try:
                raw_file_path=os.path.join(raw_file_path,file)
                df=pd.read_csv(raw_file_path)
                return df
            
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")

        logging.warning(f"No CSV file found in {raw_file_path}.")
        return None
        

        










def main():
    df=read_data()
    # print(config['data']['raw'])


if __name__ == "__main__":
    main()