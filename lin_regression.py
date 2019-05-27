#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:25:04 2019

@author: dhriti
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model, metrics

dataset=pd.read_csv('/home/dhriti/Downloads/iris.csv')
mdataset=pd.read_csv('/home/dhriti/Downloads/iris1.csv')

attr1=mdataset.iloc[:20,[0]].values
#attr1---Y (dependent)
#attr2---X (independent)

attr1_train=dataset.iloc[20:,[0]].values

attr2_train=dataset.iloc[20:,[2]].values

attr1_train=attr1_train.reshape(len(attr1_train),1)
attr2_train=attr2_train.reshape(len(attr2_train),1)


attr2_test=dataset.iloc[:20,[2]].values


regr=linear_model.LinearRegression()
regr.fit(attr2_train,attr1_train)

attr1_predict=regr.predict(attr2_test)

plt.scatter(attr2_train,attr1_train,color='blue')
plt.scatter(attr2_test,attr1_predict,color='black')
plt.plot(attr2_train,regr.predict(attr2_train),linewidth=3,color='red')


deviation=[]
for i in range(0,19):
    deviation.append(abs(attr1_predict[i]-attr1[i]))
   
print("Linear Regression");
print(deviation)
