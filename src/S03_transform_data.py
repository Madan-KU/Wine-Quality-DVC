import os
import pickle
import logging
import pandas as pd

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

from sklearn.preprocessing import StandardScaler, LabelEncoder
    
config = read_config('params.yaml')


def transform_data(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    """Storing scaler as pickle file for utilization in prediction service""" 
    scaler_filename = "scaler.pkl"
    scaler_path = os.path.join(config['scaler_dir'], scaler_filename)   
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)

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