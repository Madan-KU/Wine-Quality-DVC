import os
import pandas as pd
import logging


def read_data(directory):
    if not os.path.exists(directory):
        logging.warning(f"Directory {directory} does not exist.")
        return None
    
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            try:
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)
                logging.info(f"Successfully read {file}, shape {df.shape}")
                return df, file
            
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue

    logging.warning(f"No CSV file found in {directory}.")