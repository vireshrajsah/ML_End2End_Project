import os
import sys
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.logger import logging
from src.exceptions import CustomException

def read_any():
    pass

def save_object(file_path, obj):
    '''
    Takes object, destination path and serializes the object in a pickle file in the destination path
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info ("Exception occuer in save_object at utils.py")
        raise CustomException(e,sys)
    
def model_score(true,pred):
    '''
    Takes true value and model predicted values and returns model scores mse, mae and r2 score
    '''
    try:
        mse=mean_squared_error(true,pred)
        mae=mean_absolute_error(true,pred)
        r2 =r2_score(true,pred)
        return mse, mae, r2
    except Exception as e:
        logging.info("Exception occured in model_score at utils.py")
        raise CustomException(e, sys)

def model_build_predict(algorithm, X_train, X_test, y_train):
    '''
    Takes algorithm name, train and test values and returns predicted values as well as intercep and coefficient
    '''
    try:
        model = algorithm
        model.fit(X_train,y_train)
        y_pred_train=model.predict(X_train) # prediction on train
        y_pred_test= model.predict(X_test) # prediction on test
        intercept = model.intercept_
        coeff = model.coef_

        return y_pred_train, y_pred_test, intercept, coeff
    
    except Exception as e:
        logging.info("Exception occured in model_build_predict at utils.py")
        raise CustomException(e, sys)

def evaluate_model(xtrain, xtest, ytrain, ytest, algos:dict)->dict:
    '''
    Takes train and test data sets along with dictionary of algorithm_name: algorithm(obj), returns dictionary of algorithm:r2 score
    '''
    try:
        logging.info("Initiate models evaluation in evaluate_models at utils.py")
        results = dict()
        for name, algo in algos.items():
            ypred_train, ypred_test,_,_ = model_build_predict(algo, xtrain, xtest, ytrain)
            _,_,r2_train = model_score(ytrain, ypred_train)
            _,_,r2_test = model_score (ytest, ypred_test)

            results[name] = r2_test

        logging.info("Model evaluation report generated in evaluate_model at utils.py")
        return results
    
    except Exception as e:
        logging.info("Exception occured in evaluate_model at utils.py")
        raise CustomException(e, sys)

def print_report(model_evaluation_report, best_model_name, best_model_score):
    '''
    Takes model:r2 score dictionary, best model name and best model score and prints model evaluation report.
    '''
    try:
        # Model evaluation report
        HEADER = 'MODEL EVALUATION REPORT'
        print('\n',f"{HEADER:=^50s}",'\n')
        pprint(model_evaluation_report)
        print('\n',"-"*30,'\n')
        logging.info(f"Model Report: {model_evaluation_report}")
        print (f"Best model: {best_model_name}, R2_score: {best_model_score}")
        logging.info(f"Best model: {best_model_name}, R2_score: {best_model_score}")
        print('\n', '='*50)

    except Exception as e:
        logging.info("Exception occured in print_report at utils.py")
        raise CustomException(e, sys)
    
def load_object(file_path):
    '''
    Loads pickle file and returns the object
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            pickled_object = pickle.load(file_obj)
            return pickled_object
    except Exception as e:
        logging.info("Exception occured at load_object at utils.py")
        raise CustomException(e,sys)
