import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures



data_train = pd.read_csv('T-R-train.csv')
data_test = pd.read_csv('T-R-test.csv')
#define x_train and y_train
x_train = data_train['T']
y_train = data_train['rate']
fig1 = plt.figure(figsize=(5,5))
plt.scatter(x_train,y_train)
plt.title('raw data')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()

#linear regression model prediction
x_train = np.array(x_train).reshape(-1,1)

lr1 = LinearRegression()
lr1.fit(x_train,y_train)

x_test = data_test['T']
y_test = data_test['rate']
x_test = np.array(x_test).reshape(-1,1)
y_train_predict = lr1.predict(x_train)
y_test_predict = lr1.predict(x_test)
r2_train = r2_score(y_train,y_train_predict)
r2_test = r2_score(y_test,y_test_predict)

#generate the new data
x_range = np.linspace(40,90,300).reshape(-1,1)
y_range_predict = lr1.predict(x_range)
fig2 = plt.figure(figsize=(5,5))
plt.plot(x_range,y_range_predict)
plt.scatter(x_train,y_train)
plt.title('prediction data')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()

#polynomial model

poly2 = PolynomialFeatures(degree=2)        #degree=5
x_2_train = poly2.fit_transform(x_train)
x2_test = poly2.transform(x_test)
lr2 = LinearRegression()
lr2.fit(x_2_train,y_train)
y2_train_predict = lr2.predict(x_2_train)
y2_test_predict = lr2.predict(x2_test)
r2_2_train = r2_score(y_train,y2_train_predict)
r2_2_test = r2_score(y_test,y2_test_predict)


#generate new data

x2_range = np.linspace(40,90,300).reshape(-1,1)
x2_range = poly2.transform(x2_range)
y2_range_predict = lr2.predict(x2_range)

fig3 = plt.figure(figsize=(5,5))
plt.plot(x_range,y2_range_predict)
plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test)
plt.title('prediction data')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()










