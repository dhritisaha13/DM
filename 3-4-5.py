#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:38:15 2019

@author: dhriti
"""

from sklearn import datasets
from sklearn import preprocessing
from statistics import mean, median
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
dataset = pd.read_csv('/home/dhriti/Downloads/iris1.csv')
x = dataset.iloc[:, [0]].values
max_x=max(x)
min_x=min(x)
max_x=math.ceil(max_x)
min_x=math.floor(min_x)
sub=max_x-min_x
print(sub)
if(sub==3 or sub==6 or sub==9 or sub==7):
     b1=[]
     b2=[]
     b3=[]
     for i in range(149):
         if(x[i,0]>=min_x and x[i,0]<min_x+sub/3):
              b1.append(x[i,0])
         if(x[i,0]>=min_x+sub/3 and x[i,0]<min_x+sub/3*2):
              b2.append(x[i,0])
         if(x[i,0]>=min_x+sub/3*2 and x[i,0]<=max_x):
             b2.append(x[i,0])       
     
if(sub==4 or sub==8 or sub==2):
     b1=[]
     b2=[]
     b3=[]
     b4=[]
     for i in range(149):
         if(x[i,0]>=min_x and x[i,0]<min_x+sub/4):
              b1.append(x[i,0])
         if(x[i,0]>=min_x+sub/4 and x[i,0]<min_x+sub/4*2):
              b2.append(x[i,0])
         if(x[i,0]>=min_x+sub/4*2 and x[i,0]<min_x+sub/4*3):
             b3.append(x[i,0])  
         if(x[i,0]>=min_x+sub/4*3 and x[i,0]<=max_x):
             b4.append(x[i,0])
if(sub==1 or sub==5 or sub==10):
     b1=[]
     b2=[]
     b3=[]
     b4=[]
     b5=[]
     for i in range(149):
         if(x[i,0]>=min_x and x[i,0]<min_x+sub/5):
              b1.append(x[i,0])
         if(x[i,0]>=min_x+sub/5 and x[i,0]<min_x+sub/5*2):
              b2.append(x[i,0])
         if(x[i,0]>=min_x+sub/5*2 and x[i,0]<min_x+sub/5*3):
             b3.append(x[i,0])  
         if(x[i,0]>=min_x+sub/5*3 and x[i,0]<min_x+sub/5*4):
             b4.append(x[i,0])
         if(x[i,0]>=min_x+sub/5*4 and x[i,0]<=max_x):
             b5.append(x[i,0])