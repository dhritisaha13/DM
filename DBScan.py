#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:26:09 2019

@author: dhriti
"""

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from collections import Counter

#Reading the Dataset
dataset = pd.read_csv("/home/dhriti/Downloads/iris1.csv")

#Taking 4 Attributes Only
x = dataset.iloc[:, [0,1, 2, 3]].values
model= DBSCAN(eps=0.8,min_samples=19).fit(x)
print(model)
outliers_df=pd.DataFrame(x)
print(Counter(model.labels_))
print (outliers_df[model.labels_==-1])
fig = plt.figure()
ax = Axes3D(fig)
#ax=fig.add_axes([1,1,1,1])
colours = model.labels_
ax.scatter(x[:,2],x[:,1],x[:,3],c=colours)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Sepal Length')
plt.title('DBSCAN for Outlier Detection')