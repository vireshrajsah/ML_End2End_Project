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
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info ("Exception occuer in save_object at utils.py")
        raise CustomException(e,sys)
    
def model_score(true,pred):
    try:
        mse=mean_squared_error(true,pred)
        mae=mean_absolute_error(true,pred)
        r2 =r2_score(true,pred)
        return mse, mae, r2
    except Exception as e:
        logging.info("Exception occured in model_score at utils.py")
        raise CustomException(e, sys)

def model_build_predict(algorithm, X_train, X_test, y_train):
    try:
        model = algorithm()
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