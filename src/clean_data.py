import os
import logging
# import argparse
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer


# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


config = read_yaml_config('parameters.yaml')

# Configuring logging using YAML values
logging.basicConfig(level=config['logging']['level'],
                    format=config['logging']['format'],
                    handlers=[logging.FileHandler(config['logging']['log_file']),
                               logging.StreamHandler()])

def read_data():
    raw_file_path=config['data']['raw']

    if not os.path.exists(raw_file_path):
        logging.warning(f"Directory {raw_file_path} does not exist.")

        return None
    
    for file in os.listdir(raw_file_path):

        if file.endswith(".csv"):
            try:
                raw_file_path=os.path.join(raw_file_path,file)
                df=pd.read_csv(raw_file_path)
                logging.info(f"Successfully read {file}")
                return df,file
            
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue

    logging.warning(f"No CSV file found in {raw_file_path}.")
    return None


def clean_data(df):
    logging.info("Cleaning data...")

    target=config['base']['target_col']
    
    df.dropna(subset=[target], inplace=True)
    imputer = SimpleImputer(strategy='mean')
    cleaned_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    logging.info("Imputed missing values using SimpleImputer")

    return cleaned_df

def main():
    df,filename=read_data()
    filename='cleaned_'+ filename
    output_path = os.path.join(config['data']['cleansed'], filename)

    if df is not None:
        cleaned_data=clean_data(df)
        cleaned_data.to_csv(output_path, index=False)
        logging.info(f"'{filename}' loaded to '{config['data']['cleansed']}'")

    else:
        logging.warning("No valid data was read.") 
    

if __name__ == "__main__":
    main()