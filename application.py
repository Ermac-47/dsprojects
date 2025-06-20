from flask import Flask,request,render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app =application

#route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
    crim=float(request.form.get('crim')),
    zn=float(request.form.get('zn')),
    indus=int(request.form.get('indus')),
    chas=float(request.form.get('chas')),
    nox=float(request.form.get('nox')),
    rm=float(request.form.get('rm')),
    age=float(request.form.get('age')),
    dis=float(request.form.get('dis')),
    rad=int(request.form.get('rad')),
    tax=int(request.form.get('tax')),
    ptratio=float(request.form.get('ptratio')),
    b=float(request.form.get('b')),
    lstat=float(request.form.get('lstat')),
    medv=float(request.form.get('medv'))
    )
    pred_df=data.get_data_as_df()
    print(pred_df)

    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run()