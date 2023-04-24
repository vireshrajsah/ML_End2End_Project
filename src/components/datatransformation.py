import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

# Input arguments: train file path and test file path
# Returns: Transformed csv files and Pipeline pickle file

@dataclass
class DataTransformationConfig:
    '''
    Data class to store preprocessor.pkl path
    '''
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''
    Class Data transformation
    '''
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        Creates and returns transformation pipeline object
        '''
        try:
            logging.info("Data transformation initiated by get_data_transformation_object at datatransformation.py")
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_map=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            clarity_map=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            color_map= ['D', 'E', 'F', 'G', 'H', 'I', 'J']

            logging.info("Data-transformation pipeline initiated")
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps= [
                ('imputation', SimpleImputer(strategy='median')),
                ('scaling', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps = [
                ('imputation', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoding', OrdinalEncoder(categories=[cut_map, color_map, clarity_map])),
                ('scaling', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            logging.info("Pipeline Execution complete without any exceptions")

        except Exception as e:
            logging.info('Exception occured at get_data_transformation_object in datatransformation.py')
            raise CustomException(e,sys) from e

    def initiate_data_transformation(self, train_file_path, test_file_path):
        try:
            logging.info("Initiated data transformation on initiate_data_transformation at datatransformations.py")
            # Get train - test datasets
            train_df = pd.read_csv(train_file_path)
            logging.info("Train data loaded")
            logging.info(f"Train DF Head: \n {train_df.head().to_string()}")
            test_df = pd.read_csv(test_file_path)
            logging.info("Test data loaded")
            logging.info(f"Test DF Head: \n {test_df.head().to_string()}")

            # X-y split
            logging.info('X - y split initiated')
            target_column= 'price'
            drop_columns = ['Unnamed: 0', target_column]
            input_features_train = train_df.drop(drop_columns, axis = 1)
            target_feature_train = train_df[target_column]
            input_features_test = test_df.drop(drop_columns, axis = 1)
            target_feature_test = test_df[target_column]
            logging.info("X - y split achieved on train and test datasets")

            # Preprocessor pipeline init
            logging.info("Obtaining preprocessing object from initiate_data_transformation")
            preprocessing_obj = self.get_data_transformation_object()
            logging.info("Preprocessor object obtained at initiate_data_transformation")

            # Transformation using preprocessor obj
            logging.info("initiated preprocessing")
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)
            logging.info("preprocessing complete")

            # Obtaining train array and test array for faster operation
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test)]

            # Saving Preprocessor as pickle
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path, 
                preprocessing_obj
            )
            logging.info("Preprocessor object saved as pickle in path {0}".format(self.data_transformation_config.preprocessor_obj_file_path))

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation at datatransformation.py")
            raise CustomException(e,sys)
