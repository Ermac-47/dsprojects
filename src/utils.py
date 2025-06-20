import os
import sys
import dill
from src.exception import Custom_Exception
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    except Exception as e:
        raise Custom_Exception(e,sys) 
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report= {}
        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)

            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)

            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report 
    except Exception as e:
        raise Custom_Exception(e,sys)    
def load_object(file_path):
        try:
            with open(file_path,"rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise Custom_Exception(e,sys)
