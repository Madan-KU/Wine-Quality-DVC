import os
import pandas as pd
import logging

from modules.logger_configurator import configure_logger
from modules.read_config import read_config
from modules.data_loader import read_data


config = read_config('params.yaml')

def main():
    configure_logger()
    
    df,filename = read_data(config['data']['remote'])
    df.columns = df.columns.str.replace(' ', '_')
    # filename='raw_'+ filename

    if df is not None:
        print(df)
        
    output_path = os.path.join(config['data']['raw'], filename)
    df.to_csv(output_path, index=False)

    logging.info(f"'{filename}' loaded to '{config['data']['raw']}'")

if __name__ == "__main__":
    main()

