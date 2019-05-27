#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:42:28 2019

@author: dhriti
"""

import random
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from statistics import mean
from scipy.spatial import distance
import numpy as np

iris = datasets.load_iris()
data=[0 for x in range (150)]
total=150
for x in range(0,total):
    data[x]=random.randint(1,150)

print(data)
data.sort(key=int)
bin_no=10
size=math.ceil(total/bin_no)
       
binmean=[0 for x in range (total)]
for x in range(0,10):
    sum=0
    for y in range(0,size):
        sum=sum+data[x*size+y]
    mean1=math.ceil(sum/size)
   # print (mean1)
    for y in range(0,size):
        binmean[x*size+y]=mean1
       
#print(binmean)
        
binmedian=[0 for x in range (total)]
for x in range(0,10):
    sum=0
    median=data[x*size+math.ceil(size/2)]
    for y in range(0,size):
        binmedian[x*size+y]=median
        
#print(binmedian) 
       
binboundary=[0 for x in range (total)]
for x in range (0,10):
    for y in range(0,size):
        lb=data[x*size]
        ub=data[x*size+size-1]
        if((data[x*size+y]-lb)<=(ub-data[x*size+y])):
            binboundary[x*size+y]=lb
        else:
            binboundary[x*size+y]=ub

plt.subplot(1,3,1)
plt.plot(binmean)
plt.show()
plt.subplot(1,3,2)
plt.plot(binmedian)
plt.show()
plt.subplot(1,3,3)
plt.plot(binboundary)
plt.show()

