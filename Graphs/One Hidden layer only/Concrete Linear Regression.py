# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:31:11 2019

@author: Gemy
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import keras



# Importing the dataset

df = pd.read_csv('Concrete_Data_Yeh.csv')
x = df.drop('csMPa',axis  = 1 ).values
y = df['csMPa'].values

 ## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# 
# # Fitting Linear Regression Model
# from sklearn.linear_model import RidgeCV
# reg = RidgeCV(alphas = [0.1, 1.0, 10.0],fit_intercept =True , cv = 4)
# reg.fit(X_train,y_train)
# 
# print(reg.score(X_test,y_test))
# # Predicting On the Test Set
# y_predict = reg.predict(X_test)
# 
# 
# 
# =============================================================================


# Fitting ANN With one hidden layer
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from keras import regularizers
from keras.layers import Dropout


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

r2 = []
num = []
hist = []
hist_val = []
#Building  first layer 
model=Sequential()

for i in range(100,1100,100):
 num.append(i)
 model.add(Dense(i,input_dim=8,activation = 'relu'))



# Output Layer
 model.add(Dense(1,activation='linear'))


# Optimize , Compile And Train The Model 
 opt =keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)

 model.compile(optimizer=opt,loss='mean_squared_error',metrics=[rmse])
 history = model.fit(X_train,y_train,epochs = 35 ,batch_size=32,validation_split=0.1)
 hist.append(history.history['loss'][-1])
 hist_val.append(history.history['val_loss'][-1])
 print(model.summary())


 y_predict = model.predict(X_test)

 from sklearn.metrics import r2_score
 print(r2_score(y_test,y_predict))
 r2.append(r2_score(y_test,y_predict))

plt.plot(num,hist)
plt.plot(num,hist_val)
plt.title('Loss Vs #Neurons in one Hidden Layer ')
plt.xlabel('No.of Neurons')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('1')
plt.show()



plt.plot(num,r2)
plt.title('Root Mean Square Vs #Neurons ')
plt.xlabel('No.of Neurons')
plt.ylabel('R^2')
plt.savefig('2')
plt.show()















