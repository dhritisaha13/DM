#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:16:42 2019

@author: dhriti
"""


import matplotlib 
#matplotlib.use('GTKAgg') 

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model 
import pandas as pd 

df = pd.read_csv("/home/dhriti/Downloads/iris.csv")
df1=pd.read_csv("/home/dhriti/Downloads/iris1.csv")
#new_df=df.dropna(axis = 0, how ='any')

X = df.iloc[:, [0]].values
Y= df.iloc[:, [2]].values
#Y = df['C'] 
#X = df['A'] 
   
X=X.reshape(len(X),1) 
Y=Y.reshape(len(Y),1) 
   
# Split the data into training/testing sets 
X_train = X[:-30] 
X_test = X[-30:] 
   
# Split the targets into training/testing sets 
Y_train = Y[:-30] 
Y_test = Y[-30:] 
   
# Plot outputs 
plt.scatter(X_test, Y_test,  color='black') 
plt.title('Test Data') 
plt.xlabel('A') 
plt.ylabel('C') 
plt.xticks(()) 
plt.yticks(()) 
   
  
# Create linear regression object 
regr = linear_model.LinearRegression() 
   
# Train the model using the training sets 
regr.fit(X_train, Y_train) 
   
# Plot outputs 
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3) 
plt.show() 
#print(Y_test)
#print(Y_train)
#print(regr.predict(X_test))
Y1= df1.iloc[:, [2]].values
Y1_test = Y1[-30:]

for i in range(30):
     if Y_test[i]==0:
         print(Y1_test[i]+regr.predict(X_test[i])+(regr.predict(X_test[i])-regr.predict(Y1_test[i])))
print('$')

a=((Y1_test)-regr.predict(X_test))
b=regr.predict(X_test)

table={}
for j in range(30):
    table[j]=[Y1_test[j],b[j],a[j]]
print(table)   
 










 


