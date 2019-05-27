#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 18:08:38 2019

@author: dhriti
"""


from sklearn import datasets
from sklearn import preprocessing
from statistics import mean,median
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

#importing dataset
iris = datasets.load_iris()
X=iris.data

#normalisation
scalar = preprocessing.MinMaxScaler()
scalar.fit(X)
normalized = scalar.transform(X)
#print(normalized[0])

#calculating distance
euclidean=[]
for i in range(150):
    euclidean.append([])
    a=normalized[i]
    for j in range(150):
       # print(euclidean)
       # a=normalized[i]
        b=normalized[j]
        dst = distance.euclidean(a, b)
        #print(dst)
        euclidean[i].append(dst)
#print(euclidean)   

#cluster formation
sets=[]

for i in range(150):
    xm = mean(euclidean[i])
    #print(xm)
    ss=set([])
    for j in range(len(euclidean[i])):
        if euclidean[i][j]<xm:
            ss.add(j)
    sets.append(ss)
#print(sets[1])    

def jaccard(x,y):
    return len(x&y)/len(x|y)

#subset removal
while len(sets)>3:
    setsc=set([])
    
    for i in range(len(sets)):
        for j in range(i,len(sets)):
            if sets[i]>=sets[j] and i!=j:
                setsc.add(j)
    #print((setsc))
    xx=[]
    for i in setsc:
        xx.append(i)
    
    for i in  sorted(xx,reverse=True):
        sets.remove(sets[i])

    #print(len(sets),len(setsc))
    #print(len(sets))
    if len(sets)<=3:
        break

#creating similarity matrix    
    setsim=[]

    for i in range(len(sets)):
        x=[]
        for j in range(len(sets)):
            if j>i:
                x.append(jaccard(sets[i],sets[j]))
            else:
                x.append(0)
        setsim.append(x)
    
    ma=0
    mi=0
    mj=0
    
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            if ma<setsim[i][j]:
                ma=setsim[i][j]
                mi=i
                mj=j
    
    if ma<0.25:
        break
   
#merging the clusters
    c=sets[mi]|sets[mj]
    #print(c)
    if mi>mj:
        sets.remove(sets[mi])
        sets.remove(sets[mj])
    else:
        sets.remove(sets[mj])
        sets.remove(sets[mi])
    sets.append(c)
    #print([len(i) for i in sets],len(sets))

#print([len(i) for i in sets])

import numpy as np
import matplotlib.pyplot as plt
x1=[]
x2=[]
x3=[]
s1=set()
s2=set()
s3=set()
c=0
#print(len(sets))
for i in sets:
    for j in i:
        if c==0:
            x1.append(X[j])
            s1.add(j)
            
        if c==1:
            x2.append(X[j])
            s2.add(j)
            
        if c==2:
            x3.append(X[j])
            s3.add(j)
            
    c+=1





x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

c11=median(x1[:,0])
c12=median(x1[:,1])
c13=median(x1[:,2])
c14=median(x1[:,3])
c1=[c11,c12,c13,c14]
#print(c1)

c21=median(x2[:,0])
c22=median(x2[:,1])
c23=median(x2[:,2])
c24=median(x2[:,3])
c2=[c21,c22,c23,c24]
#print(c2)

c31=median(x3[:,0])
c32=median(x3[:,1])
c33=median(x3[:,2])
c34=median(x3[:,3])
c3=[c31,c32,c33,c34]
#print(c3)

table={}
#all 3 
i1=s1.intersection(s2,s3)
#print(i1)
for j in i1:
    dc1=distance.euclidean(X[j],c1)
    dc2=distance.euclidean(X[j],c2)
    dc3=distance.euclidean(X[j],c3)
    #print(j,' ',round(dc1/(dc1+dc2+dc3),5),' ',round(dc2/(dc1+dc2+dc3),5),' ' ,round(dc3/(dc1+dc2+dc3),5))
    table[j]=[round(dc1/(dc1+dc2+dc3),5),round(dc2/(dc1+dc2+dc3),5),round(dc3/(dc1+dc2+dc3),5)]
#print("=====")
#print(table)


i20=(s2&s3)
#print(len(i20))
#print(len(i1))
i2=i20.difference(i1)
#print(len(i2))
for j in i2:
    dc2=distance.euclidean(X[j],c2)
    dc3=distance.euclidean(X[j],c3)
    #print(j,' ','--',' ',round(dc2/(dc3+dc2),5),' ',round(dc3/(dc2+dc3),5))
    table[j]=['--',round(dc2/(dc3+dc2),5),round(dc3/(dc2+dc3),5)]

#print(table)
i30=(s3&s1)
i3=i30.difference(i1)
#print(len(i3))
for j in i3:
    dc1=distance.euclidean(X[j],c1)
    dc3=distance.euclidean(X[j],c3)
    #print(j,' ',round(dc1/(dc3+dc1),5),' ','--',' ',round(dc3/(dc1+dc3),5))
    table[j]=[round(dc1/(dc3+dc1),5),'--',round(dc3/(dc1+dc3),5)]

#print(table)
for key in sorted(table.keys()):
    print( "%s: %s" % (key, table[key]))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.scatter(x1[:,3],x1[:,0],x1[:,2],c='red',s=x1[:,1]*10)
ax.scatter(x2[:,3],x2[:,0],x2[:,2],c='blue',s=x2[:,1]*10)
ax.scatter(x3[:,3],x3[:,0],x3[:,2],c='green',s=x3[:,1]*10)
#ax.scatter(centroids[:,3],centroids[:,0],centroids[:,2],c='y',s = 50,marker='*')

ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')


plt.show()

