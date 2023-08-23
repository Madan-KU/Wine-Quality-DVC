import os
import logging
import argparse
import pandas as pd
import yaml

from modules.data_loader import read_data
from modules.logger_configurator import configure_logger


# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
config = read_yaml_config('parameters.yaml')
# print(f"config.yaml:{config}\n")


def main():
    configure_logger()
    
    df,filename = read_data(config['data']['remote'])
    # filename='raw_'+ filename

    if df is not None:
        print(df)
        
    output_path = os.path.join(config['data']['raw'], filename)
    df.to_csv(output_path, index=False)

    logging.info(f"'{filename}' loaded to '{config['data']['raw']}'")

if __name__ == "__main__":
    main()

