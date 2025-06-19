import os
import sys
from src.logger import logging
from src.exception import Custom_Exception
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd 
import numpy as np

from sklearn.compose import ColumnTransformer

from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):

        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns=[
                'crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat'
            ]
            
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scalar",StandardScaler())     
                #similary cat_pipeline ,strategy -> most_frequent
                #and one hot encoder 
                ]
            )

            logging.info(f"numerical columns standard scaling completed :{numerical_columns}")
            #logging.info("categorial columns standard scaling completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)
                #("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise Custom_Exception(e,sys)            

    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completely")
            logging.info("obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_column_name="medv"

            numerical_columns=[
                'crim',	'zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']


            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe ")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object")\
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise Custom_Exception(e,sys)
