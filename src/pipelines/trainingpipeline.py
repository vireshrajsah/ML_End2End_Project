import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
from src.components.dataingestion import DataIngestion
from src.components.datatransformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import *

from warnings import filterwarnings
filterwarnings('ignore')

# Run training pipeline
if __name__=="__main__":
    obj = DataIngestion()
    trainpath, testpath = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(trainpath, testpath)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
