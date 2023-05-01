import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exceptions import CustomException
from src.utils import *


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("PredictPipeline.predict initiated at predictionpipeline.py")
            preprocessor_path = os.path.join ('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Loading preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info("Preprocessor and model objects loaded from artifacts")
            logging.info(f"Model metrics: intercept = {model.intercept_}; coefficients= \n {model.coef_}")

            # Data Preprocessing
            data_scaled = preprocessor.transform(features)
            logging.info("Data scaled and transformed using preprocessor")
            logging.info(data_scaled)

            # Prediction
            predicted_value = model.predict(X=data_scaled)
            logging.info("Predicted value/s obtained")

            return predicted_value

        except Exception as e:
            logging.info("Exception occured in predict at predictionpipeline.py")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
    
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
        logging.info("get_data_as_dataframe initiated at predictionpipeline.py")
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Inputs gathered in dataframe")

            return df
        
        except Exception as e:
            logging.info("Exception occured in get_data_as_dataframe at predictionpipeline.py")
            raise CustomException(e,sys)
