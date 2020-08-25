import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt




data = pd.read_csv('iris.csv')
data = data.drop('MyUnknownColumn',axis=1)
x = data.drop(['Species','label'], axis = 1)
y = data['label']

#establish the decision tree model
dc_tree = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)
dc_tree.fit(x,y)

#evaluate the model
y_predict = dc_tree.predict(x)
accuracy = accuracy_score(y,y_predict)
fig1 = plt.figure(figsize=(10,10))
tree.plot_tree(dc_tree,filled=True,feature_names=['Sepallength','Sepalwidth','Petallength','Petalwidth'],class_names=['setosa','versicolor','virginica'])












