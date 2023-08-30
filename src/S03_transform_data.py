# --- NOTES ---
# 1. Reads cleansed dataset
# 2. Transforms using StandardScaler
# 3. Scales both X and y
#    - Beneficial for regression, neural nets
#    - Inverse-transform for interpretability
# 4. Saves scalers as pickle for inference
# -------------

import os
import pickle
import logging
import pandas as pd

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

from sklearn.preprocessing import StandardScaler, LabelEncoder
    
config = read_config('params.yaml')


def transform_data(data,prefix):
    """Storing scaler as pickle file for utilization in prediction service"""
    try:
        scaler = StandardScaler()
        
        # If the data is a Series (1D), reshape it to 2D
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
            is_series = True
        else:
            is_series = False

        data_scaled = scaler.fit_transform(data)
        
        # Convert back to DataFrame or Series based on the input
        if is_series:
            data_scaled = pd.Series(data_scaled.ravel())
        else:
            data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        scaler_filename = prefix+"_"+"scaler.pkl"
        scaler_path = os.path.join(config['scaler_dir'], scaler_filename)   
        
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)

        logging.info(f"Scaler saved at {scaler_path}")
        return data_scaled

    except Exception as e:
        logging.error(f"Error occurred during transformation: {e}")
        return None



def split_data(df, target):
    try:
        X = df.drop(columns=target)
        y = df[target]
        return X, y

    except Exception as e:
        logging.error(f"Error occurred during splitting data: {e}")
        return None, None


def save_data(filename, output_path, data):
    try:
        data.to_csv(output_path, index=False)
        logging.info(f"'{filename}' saved to '{output_path}'")
        return True

    except Exception as e:
        logging.error(f"Error occurred while saving '{filename}' to '{output_path}': {e}")


def main():
    configure_logger()
    target = config['base']['target_col']

    try:
        df, filename = read_data(config['data']['cleansed'])
        logging.info(f"Data read from {config['data']['cleansed']}")
        
        X, y = split_data(df, target)

        if X is not None and y is not None:

            transformed_X = transform_data(X,"X")
            transformed_y = transform_data(y,"y")
        
            if transformed_X is not None and transformed_y is not None:

                X_output_path = os.path.join(config['data']['transformed']['X'], filename)
                y_output_path = os.path.join(config['data']['transformed']['y'], filename)

                if not os.path.exists(config['data']['transformed']['X']):
                    os.makedirs(config['data']['transformed']['X'])

                if not os.path.exists(config['data']['transformed']['y']):
                    os.makedirs(config['data']['transformed']['y'])
            
                save_X_flag= save_data(filename, X_output_path, X)
                save_y_flag=save_data(filename, y_output_path, y)
                if save_X_flag and save_y_flag:
                    logging.info(f"Data split and saved to: X -> '{X_output_path}', y -> '{y_output_path}'")

            else:
                logging.warning("Data transformation failed")
        else:
            logging.warning("Failed to split data into X and y")

    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")


if __name__ == "__main__":
    main()
