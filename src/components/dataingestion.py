import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exceptions import CustomException
from src.components.datatransformation import DataTransformation

## Initialize Data ingestion configuration

# No input arguments: Raw data is fetched from predefined path
# Operation: Creates a raw csv backup, a train csv and a test csv
# Returns: train data path, test data path

@dataclass
class DataIngestionConfig:
    '''
    Data class storing train, test and raw file paths
    '''
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

## Class for Data Ingestion
class DataIngestion:
    '''
    Class Data ingestion
    '''
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Takes no arguments, returns train and test data paths
        '''
        logging.info('Data ingestion initiated at dataingestion.py')
        try:
            df:pd.DataFrame = pd.read_csv(os.path.join('dataset', 'gemstones', 'cubic_zirconia.csv'))
            logging.info('Dataset read as pandas DataFrame')

            RAW_DATA_PATH_TREE=os.path.dirname(self.ingestion_config.raw_data_path)
            os.makedirs(RAW_DATA_PATH_TREE, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index= False)
            logging.info("Raw csv created")

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.30)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index =False, header = True)
            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occured at initiate_data_ingestion")
            raise CustomException(e, sys) from e
