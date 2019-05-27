#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:16:42 2019

@author: dhriti
"""


import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import linear_model
import pandas as pd
import numpy as np
dataset = pd.read_csv('/home/dhriti/Downloads/iris1.csv')
x = dataset.iloc[:, [0,1,2,3]].values

X1= dataset.iloc[:, [0]].values[:135]
X2= dataset.iloc[:, [1]].values[:135]
X3= dataset.iloc[:, [2]].values[:135]
X4= dataset.iloc[:, [3]].values[:135]
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)

df = DataFrame(x,columns=['X1','X2','X3','X4'])


X = df[['X1','X2']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['X4']


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
#print(regr.coef_[0])

# prediction with sklearn
#New_Interest_Rate = 2.75
#New_Unemployment_Rate = 5.3
x11=dataset.iloc[:, [0]].values[-15:]
x22=dataset.iloc[:,[1]].values[-15:]
#print(x22)
output=[]
for i in range(15):
  k=x11[i]
  q=x22[i]

  output.append(regr.coef_[0]*k+regr.coef_[1]*q+regr.intercept_)
print(output)
import matplotlib.pyplot  as plt
x=[i for i in range (0,5)]
plt.plot(x,x,color='green')
plt.scatter(output,dataset.iloc[:, [3]].values[-15:],color='orange')
plt.show()

