import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.logger import logging
from src.exceptions import CustomException
from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training (self, train_array, test_array):
        try:
            logging.info("Initiated model training on ModelTrainer at model_trainer.py")

            # X -y split
            xtrain, xtest, ytrain, ytest = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            # List of algorithms to be tried on the dataset
            algo_dict = {'LinearRegression':LinearRegression,
                        'Ridge':Ridge, 
                        'Lasso':Lasso, 
                        'ElasticNet':ElasticNet}
            
            # Evaluation of algorithms on the dataset
            logging.info("Models sent for evaluation in initiate_model_training at model_trainer.py")
            model_evaluation_report:dict = evaluate_model(xtrain, xtest, ytrain, ytest, algo_dict)
            logging.info("Model evaluation report returned in initiate_model_training at model_trainer.py")

            # Best model and algorithm
            best_model_score = max (model_evaluation_report.values())
            best_model_name = max (model_evaluation_report, key=model_evaluation_report.get)
            best_algo = algo_dict[best_model_name]

            # Print report
            logging.info("Sending for report generation from model_trainer.py")
            print_report(model_evaluation_report, best_model_name, best_model_score)
            logging.info("Report generated and returned")

            # Saving best model
            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj= best_algo
            )
            logging.info("Best model saved as pickle")

        except Exception as e:
            logging.info("Exception occured in ModelTrainer at model_trainer.py")
            raise CustomException(e,sys)