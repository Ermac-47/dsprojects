import os
import sys
from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_obj,evaluate_model

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split training and test input data")

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "Catboosting Regressor":CatBoostRegressor(),
                "Adaboost Regressor":AdaBoostRegressor()
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            # best model score from dict
            best_model_score=max(sorted(model_report.values()))

            # best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise Custom_Exception("No best model found !!")
            logging.info("Best found model on both training and test data-set")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted_output=best_model.predict(x_test)

            r2_scores=r2_score(y_test,predicted_output)

            return r2_scores


        except Exception as e:
            raise Custom_Exception(e,sys)
