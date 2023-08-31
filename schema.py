import json
import logging
from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

configure_logger()


def main():
    config = read_config('params.yaml')
    schema_path = config['schema']['input']
    target_column=config['base']['target_col']
    
    try:
        
        df, file = read_data(config['data']['cleansed'])
        
        df_describe = df.drop(columns=target_column).describe()
        df_describe_json = df_describe.loc[["min","max"]].to_json()

    

        with open(schema_path, "w+") as file:
            file.write(df_describe_json)
            logging.info(f"Schema written to '{schema_path}'")
    except Exception as e:
        logging.error(f"Unable to write input schema. Error: {e}")

if __name__ == "__main__":
    main()