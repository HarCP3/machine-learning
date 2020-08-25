import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

data = pd.read_csv('generated.csv')

x = data.loc[:,'x']
y = data.loc[:,'y']
x = np.array(x)
x = x.reshape(-1,1)
y = np.array(y)
y = y.reshape(-1,1)
lr_model = LinearRegression()
lr_model.fit(x,y)
y_predict = lr_model.predict(x)
#a,b 打印
a = lr_model.coef_
b = lr_model.intercept_
MSE = mean_squared_error(y,y_predict)
R2 = r2_score(y,y_predict)



