import sys
import os
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object
import dill
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:

            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise Custom_Exception(e,sys)







class CustomData:
    def __init__(self,
                 crim:float,
                 zn	:float,
                 indus:int,
                 chas:float,	
                 nox:float,	
                 rm	:float,
                 age:float,	
                 dis:float,	
                 rad:int,	
                 tax:int,
                 ptratio:float,
                 b:float,
                 lstat:float,	
                 medv:float):
        
        self.crim = crim
        self.zn = zn
        self.indus = indus
        self.chas = chas
        self.nox = nox
        self.rm = rm
        self.age = age
        self.dis = dis
        self.rad = rad
        self.tax = tax
        self.ptratio = ptratio
        self.b = b
        self.lstat = lstat
        self.medv = medv

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "crim": [self.crim],
                "zn": [self.zn],
                "indus": [self.indus],
                "chas": [self.chas],
                "nox": [self.nox],
                "rm": [self.rm],
                "age": [self.age],
                "dis": [self.dis],
                "rad": [self.rad],
                "tax": [self.tax],
                "ptratio": [self.ptratio],
                "b": [self.b],
                "lstat": [self.lstat],
                "medv": [self.medv],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Custom_Exception(e,sys)