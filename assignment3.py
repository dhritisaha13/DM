#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:03:55 2019

@author: dhriti
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

dataset =  pd.read_csv("/home/dhriti/Downloads/train.csv")
#print(dataset)
data= dataset.iloc[:,1:-1].values
m= data.shape[0]
data=np.array(data)

n= data.shape[1]

cor_mat= np.zeros((n,n))


def pearson_correlation(i,j):
    x= data[:,i]
    y= data[:,j]
    x_m= np.mean(x)
    y_m= np.mean(y)
    num=0
    den1=0
    den2=0
    for i in range(0,m):
        num+=(x[i]-x_m)*(y[i]-y_m)
        den1+= (x[i]-x_m)*(x[i]-x_m)
        den2+= (y[i]-y_m)*(y[i]-y_m)
    den1= math.sqrt(den1)
    den2= math.sqrt(den2)
    return num/(den1*den2)

for i in range(n):
    for j in range(n):
        cor_mat[i][j]= pearson_correlation(i,j)
        
select_edge= np.ones((n,n))
no_of_edges_left=n*n

# consider the two-dimensional matrix cor_mat as the adjacency matrix for the graph, we filter out all the edges which lie below a particular threshold
        
thresh = 0.2

for i in range(n):
    for j in range(n):
        if i==j:
            select_edge[i][j]=0
            no_of_edges_left-=1
            continue
        if cor_mat[i][j]<thresh:
            select_edge[i][j]=0
            no_of_edges_left-=1

select_features=[]

while no_of_edges_left>0:
    most_important_node_val=-10000000
    most_important_node_index=-1
    for i in range(n):
        temp_sum=0
        for j in range(n):
            if select_edge[i][j]==1:
                temp_sum+=cor_mat[i][j]
        if temp_sum>most_important_node_val:
            most_important_node_val=temp_sum
            most_important_node_index=i
    
    select_features.append(most_important_node_index)
    for i in range(n):
        if select_edge[i][most_important_node_index]==1:
            select_edge[i][most_important_node_index]=0
            no_of_edges_left-=1
        if select_edge[most_important_node_index][i]==1:
            select_edge[most_important_node_index][i]=0
            no_of_edges_left-=1
    
print (select_features)
    