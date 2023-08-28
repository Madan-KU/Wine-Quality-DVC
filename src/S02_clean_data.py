import os
import logging
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from modules.data_loader import read_data
from modules.logger_configurator import configure_logger

# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


config = read_yaml_config('params.yaml')


def clean_data(df):
    logging.info("Cleaning data...")

    target=config['base']['target_col']
    
    df.dropna(subset=[target], inplace=True)
    imputer = SimpleImputer(strategy='mean')
    cleaned_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    logging.info("Imputed missing values using SimpleImputer")

    return cleaned_df

def main():
    configure_logger()

    df,filename=read_data(config['data']['raw'])
    # filename='cleaned_'+ filename
    output_path = os.path.join(config['data']['cleansed'], filename)

    if df is not None:
        cleaned_data=clean_data(df)
        cleaned_data.to_csv(output_path, index=False)
        logging.info(f"'{filename}' loaded to '{config['data']['cleansed']}'")

    else:
        logging.warning("No valid data was read.") 
    

if __name__ == "__main__":
    main()