#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:20:14 2019

@author: dhriti
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial import distance

#importing the Iris dataset with pandas
dataset = pd.read_csv('/home/dhriti/Downloads/iris.csv')
x = dataset.iloc[:, [0, 1, 2, 3]].values
#print(x)


data=np.array(x)

def normalisation(data1):

    data1 = normalize(data, axis=1, norm='max')
    print(data1)

normalisation(data)

def euclidean_distance(a,b):
    
    
        
    dist = distance.euclidean(a, b)
                  # print(dist)
    return dist
#euclidean_distance(10,30)

def similarity_matrix(data1):
    rows=np.size(data1,0)
    columns=np.size(data1,1)
    print(rows)
    print(columns)
    sim_mat=np.zeros((rows,rows))
    for i in range(rows):
        for j in range(rows):
            sim_mat[i,j]=euclidean_distance(data1[i,:],data1[j,:])
    print(sim_mat)
    print(np.shape(sim_mat))
    return sim_mat
similarity_matrix=similarity_matrix(data)     

def average(sim_mat):
    row=np.size(sim_mat,0)
    avg_mat=np.zeros((row,1))
    for i in range(row):
        avg_mat[i,0]=np.average(sim_mat[i,:])
    print(avg_mat)
    return avg_mat
avg_mat=average(similarity_matrix)  

   

'''def jaccard_similarity(x,y):
  
 intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
 union_cardinality = len(set.union(*[set(x), set(y)]))
 return intersection_cardinality/float(union_cardinality)
  
y=jaccard_similarity([0,1,2,5,6],[0,2,3,5,7,9])
#print(y)'''

'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.spatial import distance

#read data
data = np.genfromtxt('/home/dhriti/Downloads/iris.csv',delimiter=',', missing_values=0,skip_header=1, dtype=float)
#d = pd.read_csv('iris.csv').as_matrix()
#print(data)

#normalize
scaler = MinMaxScaler()
n_data = scaler.fit_transform(data)
print(n_data)

#distance
S = np.empty(shape=(len(n_data),len(n_data)))
for i in range(0,len(n_data)):
    for j in range (0,len(n_data)):
        S[i,j] = distance.euclidean(n_data[i],n_data[j])
print(S)
print(str(len(S))+' '+str(len(S[0,:])))

#initial clusters
c_arr = []
for i in range(0,len(S)):
    a = []    
    #a.append(n_data[i])
    a.append(i)
    avg = np.mean(S[i,:])
    for j in range(0,len(S[0,:])):
        if S[i,j]<avg and i!=j:
            a.append(j)
    c_arr.append(a)
#C = np.asarray(c_arr)
#print(C)  
n_c = len(c_arr)
#print(c_arr)

no_clusters = 3
while n_c > no_clusters:
#subset removal
    p = set()
    for i in range(0,len(c_arr)-1):
        for j in range(i+1,len(c_arr)):
            if set(c_arr[j]).issubset(set(c_arr[i])):
                p.add(j)
            elif set(c_arr[i]).issubset(set(c_arr[j])):
                p.add(i)
    p = list(p)
    p.sort(reverse=True)
    for x in p:
        c_arr.pop(x)
#print(c_arr[49])
    n_c = len(c_arr)
#    print(n_c)
    
#Jaccard
    C = np.empty(shape=(len(c_arr),len(c_arr)))
    for i in range(0,len(c_arr)):
        for j in range(0,len(c_arr)):
            C[i,j] = len(set(c_arr[i]) & set(c_arr[j])) / len(set(c_arr[i]) | set(c_arr[j]))
            #print(C)

#union of clusters
    for i in range(0,len(C)):
        for j in range(0,len(C)):
            if C[i,j]==1 and i==j:
                C[i,j]=-1;
    m = np.amax(C)
    print(m)
    x = []
    y = []
    for i in range(0,len(C)):
        for j in range(0,len(C)):
            if C[i,j]==m:
                x.append(i)
    i = random.choice(x)
    for j in range(0,len(C)):
        if C[i,j]==m:
            y.append(j)
    j = random.choice(y)
    c_arr[i] = list(set(c_arr[i]) | set(c_arr[j]))
    if i!=j:
        c_arr.pop(j)
        n_c -= 1 
#plt.plot(n_data)
#plt.show()

'''