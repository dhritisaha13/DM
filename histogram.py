#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:50:54 2019

@author: dhriti
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math


data=[0 for x in range (100)]
for x in range (0,100):
    data[x]=random.uniform(0,20)
   
#print(data)
i=5

hist=[0 for x in range (4)]
for m in range(0,4):
    k=0
    for x in range (0,100):
        if(data[x]>=i*m and data[x]<(m+1)*i):
            k=k+1
    hist[m]=k

print(hist)


objects = ('0-5','5-10','10-15','15-20')
y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
plt.subplot(1,2,1)
plt.bar(y_pos, hist, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
#plt.ylabel('Usage')
plt.title('Equal width')
 
plt.show()


data.sort()
binmean=[0 for x in range (4)]
for x in range(0,4):
    sum=0
    for y in range(0,5):
        sum=sum+data[x*5+y]
    mean=math.ceil(sum/5)
   # print (mean)
    binmean[x]=mean

print(binmean)
count =[5,5,5,5]
objects = binmean
y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
plt.subplot(1,2,2)
plt.bar(y_pos, count , align='center', alpha=0.5)
plt.xticks(y_pos, objects)
#plt.ylabel('Usage')
plt.title('Equal depth')
 
plt.show()