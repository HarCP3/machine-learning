import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

pd.set_option('Display.max_columns',None)
data = pd.read_csv('usa_housing_price.csv')
fig = plt.figure(figsize=(10,10))
fig1 = plt.subplot(231)
plt.scatter(data['Avg. Area Income'],data['Price'])
plt.title('Price VS Income')
fig2 = plt.subplot(232)
plt.scatter(data['Avg. Area House Age'],data['Price'])
plt.title('Price VS House Age')
fig3 = plt.subplot(233)
plt.scatter(data['Avg. Area Number of Rooms'],data['Price'])
plt.title('Price VS Number of Rooms')
fig4 = plt.subplot(234)
plt.scatter(data['Avg. Area Number of Bedrooms'],data['Price'])
plt.title('Price VS Number of Bedrooms Rooms')
fig5 = plt.subplot(235)
plt.scatter(data['Area Population'],data['Price'])
plt.title('Price VS Population')
plt.show()

# define x and y

x = data['Area Population']
y = data['Price']
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
LR1 = LinearRegression()
LR1.fit(x,y)
y_predict_1 = LR1.predict(x)
MSE1 = mean_squared_error(y,y_predict_1)
R21 = r2_score(y,y_predict_1)
fig6 = plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.plot(x,y_predict_1,'r')
plt.show()

#multi-factor model

x_multi = data.drop(['Price','Address'],axis=1)
LR_multi = LinearRegression()
LR_multi.fit(x_multi,y)
y_predict_multi = LR_multi.predict(x_multi)
MSE2 = mean_squared_error(y,y_predict_multi)
R22 = r2_score(y,y_predict_multi)
fig7 = plt.figure(figsize=(8,5))
plt.scatter(y,y_predict_multi)
plt.show()

x_test = np.array([65000,5,5,30000,200]).reshape(1,-1)
y_test_predict = LR_multi.predict(x_test)





