import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
data= pd.read_csv("models/dataset/last_covid.csv")
fd=pd.DataFrame(data)

def predict_confirm_cases(date):
    real_x = data.iloc[:, 1].values
    real_y = data.iloc[:, 2].values
    real_x = real_x.reshape(-1, 1)
    real_y = real_y.reshape(-1, 1)
    training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.3, random_state=0)
    Lin = LinearRegression()
    Lin.fit(training_x, training_y)
    pred_y = Lin.predict(testing_x)
    label=[]
    values=[]
    ls_date=list(fd['Date'].tail(1))[0]
    print(ls_date)
    input_date=datetime.datetime.strptime(ls_date, "%d/%m/%y").date()
    days=(date-input_date).days
    for i in range(0,11):
        label.append(str(date+datetime.timedelta(days=i)))
        values.append(math.ceil(Lin.predict([[40+i+days]])[0][0]))
    return label,values

def predict_recoverd_cases(date):
    real_x= data.iloc[:,1].values
    real_y= data.iloc[:,3].values
    real_x= real_x.reshape(-1,1)
    real_y= real_y.reshape(-1,1)
    training_x,testing_x,training_y,testing_y = train_test_split(real_x,real_y,test_size=0.3,random_state=0)
    Lin= LinearRegression()
    Lin.fit(training_x,training_y)
    pred_y= Lin.predict(testing_x)
    label = []
    values = []
    ls_date = list(fd['Date'].tail(1))[0]
    print(ls_date)
    input_date = datetime.datetime.strptime(ls_date, "%d/%m/%y").date()
    days = (date - input_date).days
    for i in range(0, 11):
        label.append(str(date + datetime.timedelta(days=i)))
        values.append(math.ceil(Lin.predict([[40 + i + days]])[0][0]))
    return label, values


def predict_deceased_cases(date):
    real_x= data.iloc[:,1].values
    real_y= data.iloc[:,4].values
    real_x= real_x.reshape(-1,1)
    real_y= real_y.reshape(-1,1)
    training_x,testing_x,training_y,testing_y = train_test_split(real_x,real_y,test_size=0.3,random_state=0)
    Lin= LinearRegression()
    Lin.fit(training_x,training_y)
    pred_y= Lin.predict(testing_x)
    label = []
    values= []
    ls_date = list(fd['Date'].tail(1))[0]
    print(ls_date)
    input_date = datetime.datetime.strptime(ls_date, "%d/%m/%y").date()
    days = (date - input_date).days
    for i in range(0, 11):
        label.append(str(date + datetime.timedelta(days=i)))
        values.append(math.ceil(Lin.predict([[40 + i + days]])[0][0]))
    return label, values
