import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope



data = pd.read_excel('anomaly_data.xlsx')
# plt.scatter(data['x1'],data['x2'])
# plt.title('data')
# plt.xlabel('x1')
# plt.ylabel('x2')

#define x1 x2
x1 = data['x1']
x2 = data['x2']
fig,axes = plt.subplots(1,2,figsize=(10,5))
x1.plot(kind='hist',ax = axes[0],bins=100)
x2.plot(kind='hist',ax=axes[1],bins=100)

#Gaussian distribution

x1_mean = x1.mean()
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()
x1_range = np.linspace(4,25,300)
x1_normal = norm.pdf(x1_range,x1_mean,x1_sigma)
x2_range = np.linspace(2,25,300)
x2_normal = norm.pdf(x2_range,x2_mean,x2_sigma)
fig2 = plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(x1_range,x1_normal)
plt.title('normal p(x1)')
plt.subplot(122)
plt.plot(x2_range,x2_normal)
plt.title('normal p(x2)')
plt.show()

#establish the model and predict

ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(data)
#make prediction
y_predict = ad_model.predict(data)

fig4 = plt.figure(figsize=(10,5))
original_data = plt.scatter(data['x1'],data['x2'],marker='x')
anomaly_data = plt.scatter(data['x1'][y_predict==-1],data['x2'][y_predict==-1],marker='o',facecolor='none',edgecolor='r',s=150)
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((original_data,anomaly_data),('original_data','anomaly_data'))
plt.show()









