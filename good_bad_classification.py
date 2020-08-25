import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



data = pd.read_csv('data_class_raw.csv')

#define x and y
x = data.drop('y',axis=1)
y = data['y']

# fig1 = plt.figure(figsize=(5,5))
# bad = plt.scatter(x['x1'][y==0],x['x2'][y==0],label = 'bad')
# good = plt.scatter(x['x1'][y==1],x['x2'][y==1],label='good')
# plt.title('raw data')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.legend()
# plt.show()

#anomay detection

ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(x[y==0])
y_predict_bad = ad_model.predict(x[y==0])
# ad_model.fit(x[y==1])
# y_predict_good = ad_model.predict(x[y==1])
fig2 = plt.figure(figsize=(5,5))
bad = plt.scatter(x['x1'][y==0],x['x2'][y==0],label = 'bad')
good = plt.scatter(x['x1'][y==1],x['x2'][y==1],label='good')
plt.scatter(x['x1'][y==0][y_predict_bad==-1],x['x2'][y==0][y_predict_bad==-1],marker='x',s=150)
# plt.scatter(x['x1'][y==1][y_predict_good==-1],x['x2'][y==1][y_predict_good==-1],marker='x',s=150)
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


#导入处理之后的数据
data = pd.read_csv('data_class_processed.csv')
x = data.drop('y',axis=1)
y = data['y']
x_norm = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x_norm)
var_ratio = pca.explained_variance_ratio_

fig4 = plt.figure(figsize=(5,5))
plt.bar([1,2],var_ratio)
plt.show()


#train and test split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=4,test_size=0.4)
knn_10  = KNeighborsClassifier(n_neighbors=10)
knn_10.fit(x_train,y_train)
y_train_predict = knn_10.predict(x_train)
y_test_predict = knn_10.predict(x_test)
accuracy_train = accuracy_score(y_train,y_train_predict)
accuracy_test = accuracy_score(y_test,y_test_predict)



#visualize the knn result and boundary

xx, yy = np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
x_range = np.c_[xx.ravel(),yy.ravel()]
y_range_predict = knn_10.predict(x_range)

fig5 = plt.figure(figsize=(5,5))
bad = plt.scatter(x_range[:,0][y_range_predict==0],x_range[:,1][y_range_predict==0],label = 'knn_bad')
good = plt.scatter(x_range[:,0][y_range_predict==1],x_range[:,1][y_range_predict==1],label='knn_good')
bad = plt.scatter(x['x1'][y==0],x['x2'][y==0],label = 'bad',marker='x')
good = plt.scatter(x['x1'][y==1],x['x2'][y==1],label='good')
#plt.scatter(x['x1'][y==0][y_predict_bad==-1],x['x2'][y==0][y_predict_bad==-1],marker='x',s=150)
# plt.scatter(x['x1'][y==1][y_predict_good==-1],x['x2'][y==1][y_predict_good==-1],marker='x',s=150)
plt.title('prediction result')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


cm = confusion_matrix(y_test,y_test_predict)
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
accuracy = (TP + TN)/(TP+TN+FP+FN)
recall = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)
f1_score = (2*recall*precision)/(precision+recall)



#try different k and calculate the accuracy for each


n = [i for i in range(1,21)]
accuracy_train = []
accuracy_test = []
for i in n:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_train_predict = knn.predict(x_train)
    y_test_predict = knn.predict(x_test)
    accuracy_train_i = accuracy_score(y_train,y_train_predict)
    accuracy_test_i = accuracy_score(y_test,y_test_predict)
    accuracy_train.append(accuracy_train_i)
    accuracy_test.append(accuracy_test_i)
print(accuracy_train,accuracy_test)

fig5 = plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(n,accuracy_train,marker = 'o')
plt.title('training accuracy vs n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.subplot(122)
plt.plot(n,accuracy_test,marker = 'o')
plt.title('testing accuracy vs n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.show()












