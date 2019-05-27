#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:49:47 2019

@author: dhriti
"""



w,h=150,4
Matrix = [[0 for x in range(w)] for y in range(h)]
Matrix1 = [[0 for x in range(h)] for y in range(w)]
Matrix3 = [[0 for x in range(h)] for y in range(w)]
import csv

with open('/home/dhriti/Downloads/iris1.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    y=0   
    for row in reader:
        x=0
        for x in range (h):
            Matrix[x][y]=row[x]
            #print(Matrix[x][y])
            x=x+1
        y=y+1

csvFile.close()
for i in range (h):
    for j in range (w):
        Matrix1[j][i]=Matrix[i][j]



from scipy.spatial import distance
import math
def euclidean0_1(vector1, vector2):
    s=0.0
    #dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    #dist = math.sqrt(sum(dist))
    for i in range(h):
        s=s+pow((float(vector1[i])-float(vector2[i])),2)
    dist = math.sqrt(s)
    return dist

Matrix3 = [[0 for x in range(w)] for y in range(w)]
for i in range (w):
    a=Matrix1[i]
    for j in range (w):
              b=Matrix1[j]
              d=euclidean0_1(a,b)
              if(d>.95):
                   Matrix3[i][j]=0
              else:
                   Matrix3[i][j]=1
              
              #Matrix3[i][j]=d
#print(Matrix3)

for i in range (w):
    for j in range (w):
          if(Matrix3[i][j]==1 and i!=j):
              Matrix3[j][i]=0


        
import networkx as nx
from networkx import edge_betweenness_centrality
from networkx import connected_component_subgraphs
import operator
import collections
from collections import OrderedDict


def edge_to_remove(G):
    dictl=nx.edge_betweenness_centrality(G)
    list_of_tuples=dictl.items()
    sorted_x = sorted(dictl.items(), key=lambda kv: kv[1],reverse = True)
    #sorted_dict = OrderedDict(sorted_x)
  #  print(sorted_x[0][0])
    return sorted_x[0][0]


def girvan(G):
    d=list(nx.connected_component_subgraphs(G))
    l=len(d)
 #   print("the no of conn comp")
 #   print(l)
    while(l==2 or l==1):
        G.remove_edge(*edge_to_remove(G))
        d=list(nx.connected_component_subgraphs(G))
        l=len(d)
    #    print(l)
    return d
import numpy
A=numpy.matrix(Matrix3)
G=nx.from_numpy_matrix(A)
c=girvan(G)
listc0=[]
listc1=[]
listc2=[]
listc0=c[0].nodes()
listc1=c[1].nodes()
listc2=c[2].nodes()
'''
for i in c:
    print(i.nodes())
'''
listsi=[]
def silhoutte(l1,l2,l3):
    si=0
    for i in range(150):
        if i in l1:
            a=0
            b=0
            for j in l1:
                a=a+euclidean0_1(Matrix1[i],Matrix1[j])
            for k in l2:
                b=b+euclidean0_1(Matrix1[i],Matrix1[k])
            for l in l3:
                b=b+euclidean0_1(Matrix1[i],Matrix1[l])
            s=(b-a)/max(b,a)
        if i in l2:
            a=0
            b=0
            for j in l2:
                a=a+euclidean0_1(Matrix1[i],Matrix1[j])
            for k in l1:
                b=b+euclidean0_1(Matrix1[i],Matrix1[k])
            for l in l3:
                b=b+euclidean0_1(Matrix1[i],Matrix1[l])
            s=(b-a)/max(b,a)
        if i in l3:
            a=0
            b=0
            for j in l3:
                a=a+euclidean0_1(Matrix1[i],Matrix1[j])
            for k in l2:
                b=b+euclidean0_1(Matrix1[i],Matrix1[k])
            for l in l1:
                b=b+euclidean0_1(Matrix1[i],Matrix1[l])
            s=(b-a)/max(b,a)
        si=si+s
        listsi.insert(i,s)
    sav=si/150
    return sav


sil=silhoutte(listc0,listc1,listc2)
print("silhoutte index:")
print(sil)
#print(listsi)
def intra(l):
    maxd=0
    for i in l:
        for j in l:
            d=euclidean0_1(Matrix1[i],Matrix1[j])
            if(d>maxd):
                maxd=d
    return maxd
def inter(l1,l2):
    maxd=0
    for i in l1:
        for j in l2:
            d=euclidean0_1(Matrix1[i],Matrix1[j])
            if(d>maxd):
                maxd=d
    return maxd
def dunns(l1,l2,l3):
    intram=max(intra(l1),intra(l2),intra(l3))
    inter1=min(inter(l1,l2),inter(l1,l3),100)
    inter2=min(inter(l2,l1),inter(l2,l3),100)
    inter3=min(inter(l3,l1),inter(l3,l2),100)
    interm=min(inter1,inter2,inter3)
    return interm/intram

dunn=dunns(listc0,listc1,listc2)
print("dunn index:")
print(dunn)

def db(l1,l2,l3):
    i11=(intra(l1)+intra(l2))/inter(l1,l2)
    i12=(intra(l1)+intra(l3))/inter(l1,l3)
    i1=max(i11,i12,0)
    i21=(intra(l2)+intra(l1))/inter(l1,l2)
    i22=(intra(l2)+intra(l3))/inter(l2,l3)
    i2=max(i21,i22,0)
    i31=(intra(l3)+intra(l1))/inter(l1,l3)
    i32=(intra(l2)+intra(l3))/inter(l2,l3)
    i3=max(i31,i32,0)
    return (i1+i2+i3)/3

dbi=db(listc0,listc1,listc2)
print("db index:")
print(dbi) 
