from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.metrics import accuracy_score



(x_train,y_train),(x_test,y_test) = mnist.load_data()
img1 = x_train[0]
# fig1 = plt.figure(figsize=(5,5))
# plt.imshow(img1)
# plt.title(y_train[0])

#format the input data
feature_size = img1.shape[0]*img1.shape[1]
x_train_format = x_train.reshape(x_train.shape[0],feature_size)
x_test_format = x_test.reshape(x_test.shape[0],feature_size)


#normalize the input data
x_train_normal = x_train_format/255
x_test_normal = x_test_format/255

#format output data

y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)

#set the model

mlp = Sequential()
mlp.add(Dense(units=392,activation='sigmoid',input_dim=feature_size))
mlp.add(Dense(units=392,activation='sigmoid'))
mlp.add(Dense(units=10,activation='softmax'))
mlp.summary()

#configure the model
mlp.compile(loss='categorical_crossentropy',optimizer='adam')
#train the model
mlp.fit(x_train_normal,y_train_format,epochs=10)

#evaluate the model
y_train_predict = mlp.predict_classes(x_train_normal)
accuracy_train = accuracy_score(y_train,y_train_predict)
y_test_predict = mlp.predict_classes(x_test_normal)
accuracy_test = accuracy_score(y_test,y_test_predict)


img2 = x_test[10]
fig2 = plt.figure(figsize=(5,5))
plt.imshow(img2)
plt.title(y_test_predict[10])
plt.show()





