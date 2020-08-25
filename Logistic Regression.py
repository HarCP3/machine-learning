import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



data =pd.read_csv('examdata.csv')
plt.scatter(data['Exam1'],data['Exam2'])
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.show()

#add label mask
mask = data['Pass'] ==1
passed = plt.scatter(data['Exam1'][mask],data['Exam2'][mask])
failed = plt.scatter(data['Exam1'][~mask],data['Exam2'][~mask])
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()

#define x,y
x = data.drop('Pass',axis=1)
y = data['Pass']
x1 = data['Exam1']
x2 = data['Exam2']

#establish the model and train it

LR = LogisticRegression()
LR.fit(x,y)
y_predict = LR.predict(x)
accuracy = accuracy_score(y,y_predict)
y_test = LR.predict([[70,65]])
theta0 = LR.intercept_
theta1, theta2 = LR.coef_[0][0],LR.coef_[0][1]
x2_new = -(theta0+theta1*x1)/theta2
fig3 = plt.figure()
plt.plot(x1,x2_new)
passed = plt.scatter(data['Exam1'][mask],data['Exam2'][mask])
failed = plt.scatter(data['Exam1'][~mask],data['Exam2'][~mask])
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()

#create new data
x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2
x_new = {'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
x_new = pd.DataFrame(x_new)
LR2 = LogisticRegression()
LR2.fit(x_new,y)
y2_predict = LR2.predict(x_new)
accuracy2 = accuracy_score(y,y2_predict)
LR2.coef_
x1_new = x1.sort_values()
theta0 = LR2.intercept_
theta1,theta2,theta3,theta4,theta5 = LR2.coef_[0][0],LR2.coef_[0][1],LR2.coef_[0][2],LR2.coef_[0][3],LR2.coef_[0][4]
a = theta4
b = theta5*x1_new + theta2
c = theta0 + theta1*x1_new+theta3*x1_new*x1_new
x2_new_boundary = (-b+np.sqrt(b*b-4*a*c))/(2*a)
fig5 = plt.figure()
passed = plt.scatter(data['Exam1'][mask],data['Exam2'][mask])
failed = plt.scatter(data['Exam1'][~mask],data['Exam2'][~mask])
plt.plot(x1_new,x2_new_boundary)
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()









