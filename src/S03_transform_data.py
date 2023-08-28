import os
import logging
import pandas as pd
import yaml
import sklearn
from modules.data_loader import read_data
from modules.logger_configurator import configure_logger

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Read the YAML configuration
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
config = read_yaml_config('params.yaml')


def transform_data(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def main():
    configure_logger()
    df, filename= read_data(config['data']['cleansed'])
    transformed_data =transform_data(df)
    print(transformed_data)

    output_path = os.path.join(config['data']['transformed'], filename)
    df.to_csv(output_path, index=False)

    logging.info(f"'{filename}' loaded to '{config['data']['transformed']}'")


if __name__=="__main__":
    main()