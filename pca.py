import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA




data = pd.read_csv('iris.csv')
data = data.drop('MyUnknownColumn',axis=1)
x = data.drop(['Species','label'],axis=1)
y = data['label']
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x,y)
y_predict = KNN.predict(x)
accuracy = accuracy_score(y,y_predict)

#normalize the data

x_norm = StandardScaler().fit_transform(x)

fig,axes = plt.subplots(1,2,figsize=(10,5))
x['Sepal.Length'].plot(kind='hist',bins=100,ax= axes[0])
plt.subplot(122)
plt.hist(x_norm[:,0],bins=100)
plt.show()

#pca analysis
pca = PCA(n_components=4)
x_pca = pca.fit_transform(x_norm)     #x_norm
#calculate the variance ratio of each principle components

var_ratio = pca.explained_variance_ratio_
fig2 = plt.figure(figsize=(10,5))
plt.xticks([1,2,3,4],['PC1','PC2','PC3','PC4'])
plt.bar([1,2,3,4],var_ratio)
plt.ylabel('variance ratio of each PC')
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_norm)
#visualize the PCA result
fig3 = plt.figure(figsize=(10,5))
plt.scatter(x_pca[:,0][y==1],x_pca[:,1][y==1],label = 'setosa')
plt.scatter(x_pca[:,0][y==2],x_pca[:,1][y==2],label = 'versicolor')
plt.scatter(x_pca[:,0][y==3],x_pca[:,1][y==3],label = 'virginica')
plt.legend()


KNN =KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_pca,y)
y_predict2 = KNN.predict(x_pca)
accuracy2 = accuracy_score(y,y_predict2)





