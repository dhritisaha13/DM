#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:07:18 2019

@author: dhriti
"""

def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

list1 = [.1,.5,.2]
list2 = [.2,.4,.6]

w1 = set(list1)
w2 = set(list2)
x=jaccard(w1, w2)
#print(x)

from scipy.spatial import distance
from array import array
import numpy as np
a = np.array([.1,.5,.2])
b = np.array([.2,.4,.6])
y=distance.dice(a,b)
#print(y)

from scipy import spatial

dataSetI = [.1,.5,.2]
dataSetII = [.2,.4,.6]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
#print(result)

w,h=150,4
Matrix = [[0 for x in range(w)] for y in range(h)]
Matrix1 = [[0 for x in range(h)] for y in range(w)]
Matrix2 = [[0 for x in range(h)] for y in range(w)]
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
#print(x)
#print(y)
n=y
s1=[]
s2=[]
for i in range (h):
    s1.insert(i,0.0)
    s2.insert(i,0.0)
#normal distribution
for i in range (h):
    for j in range (w):
        l=Matrix[i][j]
        #print(Matrix[j][i])
        s1[i]=s1[i]+float(l)
        s2[i]=s2[i]+(float(l)*float(l))
    p=s1[i]/n
    q=s2[i]/n
    r=pow((q-(p*p)),0.5)
  #  print("mean:")
  #  print(p)
  #  print("variance")
  #  print(r)
    for j in range (w):
        l=Matrix[i][j]
        e=(float(l)-p)/r
        f=(e+2)/4
        #print(f)
        Matrix1[j][i]=f
        #printf(Matrix1[i][j]) 
'''        
for i in range (w):
    for j in range (h):
        print(Matrix1[i][j]) 
    print("\n")
'''
#similarity matrix object vs object
Matrixs=[[0 for x in range(w)] for y in range(w)]

from scipy.spatial import distance
a = (1, 2, 3)
b = (4, 5, 6)
dst = distance.euclidean(a, b)
#print(dst)
import math
def euclidean0_1(vector1, vector2):
    s=0.0
    #dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    #dist = math.sqrt(sum(dist))
    for i in range(h):
        s=s+pow((vector1[i]-vector2[i]),2)
    dist = math.sqrt(s)
    return dist

list1=[]
list2=[]
for k in range (w):
    for j in range (h):
        list1.insert(j,Matrix1[k][j])
    for i in range (w):
        for l in range (h):
           list2.insert(l,Matrix1[i][l])
        dst = euclidean0_1(list1, list2)
        Matrixs[k][i]=dst


'''
for i in range (5):
    for j in range (5):
        print(Matrixs[i][j]) 
    print("\n")
'''

#avarage similarity
avlist=[]
for i in range (w):
    s1=0.0
    for j in range (w):
        s1=s1+Matrixs[i][j]
    s2=s1/w
    avlist.insert(i,s2)
#clusture vs object
Matrixc=[[0 for x in range(w)] for y in range(w)]

for i in range (w):
    for j in range (w):
        Matrixc[i][j]=0
for i in range (w):    
    Matrixc[i][i]=1
    for j in range (w):
            if(Matrixs[i][j]<avlist[i] and i!=j):
                Matrixc[i][j]=1


list3=[]
list4=[]

p=w
'''
for i in range (w):
    print(Matrixc[149][i])
print("HI")
'''
k=0
z=0

'''
for i in range (w):
    f=0
    list3=Matrixc[i]
    for j in range (len(list3)):
         if(list3[j]==1):
            list4.append(j)
    w1 = set(list4)
    for k in range (w):
        list5=Matrixc[k]
        for j in range (len(list5)):
              if(list5[j]==1):
                 list6.append(j)
        w2 = set(list6)
        if(w1<w2 and k!=i):
            p=p-1
            break
'''
#clusture vs object 
Matrixset=[[0 for x in range(w)] for y in range(w)]
Matrixset1=[[0 for x in range(w)] for y in range(w)]       
'''
for i in range (w):
    c=0
    list3=Matrixc[i]
    for j in range (len(list3)):
         if(list3[j]==1):
            Matrixset[i][c]=j
            c=c+1
'''
for i in range (w):
    c=0
    for j in range (w):
         if(Matrixc[i][j]==1):
            Matrixset[i][c]=j
            c=c+1

p=w
d=0

for i in range (w):
    f=0
    w1 = set( Matrixset[i])
    for k in range (w):
        w2 = set(Matrixset[k])
        if(w1<w2 and i!=k):
            p=p-1
            f=1
            break
    if(f==0):
        Matrixset1[d]=Matrixset[i]
        d=d+1
    if(f==1 and i<k):
        Matrixset1[d]=Matrixset[k]
        d=d+1
#print("number of cluster after removal of subsets:")
#print(p)


while(p!=3):
    #clusture vs clusture
    Matrixcl=[[0 for x in range(p)] for y in range(p)]
    for i in range (p):
        w1=set(Matrixset1[i])
        for j in range (p):
            w2=set(Matrixset1[j])
            r=jaccard(w1,w2)
            Matrixcl[i][j]=r

    #Matrixcr=[[0 for x in range(2)] for y in range(p)]
   
    m=0
    for i in range (p):
        for j in range (p):
            if(i !=j):
                 if(Matrixcl[i][j]>m and Matrixcl[i][j]!=1):
                     m=Matrixcl[i][j]
    
    for i in range (p):
        x=0
        for j in range (p):
            if(Matrixcl[i][j]==m):
                x=1
                break
        if(x==1):
            break     
    
    #print(i)
    #print(j)
    '''
    s=0
    f=0
    for i in range (p):
        for j in range (p):
            if(Matrixcl[i][j]==m and i!=j):
                Matrixcr[s][0]=i
                Matrixcr[s][1]=j
                f=f+1
                s=s+1

    from random import seed
    from random import randint
    def rand(f):
        seed(1)
        value = randint(0,f)
        return value
    
    if(f>0):
        r=rand(f)
        i=Matrixcr[r][0]
        j=Matrixcr[r][1]
    '''
    '''
    s=0
    Matrixclf=[[0 for x in range(p)] for y in range(p)]
    for k in range (p):
        t=0
        for l in range (p):
            if((k!=i and l!=j)and(k!=j and l!=i)):
                Matrixclf[s][t]=Matrixcl[k][l]
                t=t+1
        s=s+1
    '''
    
    w1=set(Matrixset1[i])
    w2=set(Matrixset1[j])
    w3=w1.union(w2)
    f=0
    for z in w3:
        list4.append(z)
        f=f+1
    #print(list4)
    if(i<j):
        for e in range (f):
            Matrixset1[i][e]=list4[e]
        for e in range (j,p):
            Matrixset1[e]=Matrixset1[e+1]
    else:
        for e in range (f):
            Matrixset1[j][e]=list4[e]
        for e in range (i,p):
            Matrixset1[e]=Matrixset1[e+1]
        

    list4.clear()
    p=p-1
 #   print("the number of cluster:")
  #  print(p)
   
'''
print("first cluster")
print(Matrixset1[0])
print("second cluster")
print(Matrixset1[1])
print("third cluster")
print(Matrixset1[2])
'''
listc0=[]
listc1=[]
listc2=[]
#listc3=[]
sety=set(Matrixset1[1])
for i in sety:
    listc1.append(i)
sety=set(Matrixset1[0])
sety.remove(0)
for i in sety:
    listc0.append(i)
sety=set(Matrixset1[2])
for i in sety:
    listc2.append(i)
#sety=set(Matrixset1[3])
#for i in sety:
#   listc3.append(i)

#print("first cluster")
#print(listc0)
#print("second cluster")
#print(listc1)
#print("third cluster")
#print(listc2)
#print("forth cluster")
#print(listc3)

set1=set(listc0)
set2=set(listc1)
set3=set(listc2)
#set4=set(listc3)

#print(len(listc0))
      
#print(len(listc1))

#print(len(listc2))

#print(len(listc3))
def mean(setx):
     m=[]
     k=0
     for i in range (h):
        s=0.0
        for z in (setx):
            s=s+float(Matrix[i][z])
        m.insert(k,(s/float(len(setx))))
        k=k+1
     return m
m1=[]
m2=[]
m3=[]
m1=mean(set1)
m2=mean(set2)
m3=mean(set3)


m2[0]=m2[0]-0.3
m2[2]=m2[2]-2
#print(m1)
#print(m2)
#print(m3)
def euclidean(vector1, vector2):
    s=0.0
    #dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    #dist = math.sqrt(sum(dist))
    for i in range(h):
        s=s+pow((float(vector1[i])-vector2[i]),2)
    dist = math.sqrt(s)
    return dist
Matrixb=[[0 for x in range(3)] for y in range(w)]
Matrixbf=[[0 for x in range(3)] for y in range(w)]
def min(num1, num2, num3):
    if (num1 < num2) and (num1 < num3):
        smallest_num = num1
    elif (num2 < num1) and (num2 < num3):
        smallest_num = num2
    else:
        smallest_num = num3
    return  smallest_num

for i in range (w):
    for j in range(h):
        Matrix2[i][j]=Matrix[j][i]

x=0
for i in range (w):
    if i in set1.intersection(set2,set3):
        d1=euclidean(Matrix2[i],m1)
        d2=euclidean(Matrix2[i],m2)
        d3=euclidean(Matrix2[i],m3)
        d1=d1/(d1+d2+d3)
        d2=d2/(d1+d2+d3)
        d3=d3/(d1+d2+d3)
        Matrixb[x][0]=d1
        Matrixb[x][1]=d2
        Matrixb[x][2]=d3
        d=min(d1,d2,d3)
        if(d1==d):
            Matrixbf[x][0]=1
            Matrixbf[x][1]=0
            Matrixbf[x][2]=0
            listc1.remove(i)
            listc2.remove(i)
        if(d2==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=1
            Matrixbf[x][2]=0
            listc0.remove(i)
            listc2.remove(i)
        if(d3==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=0
            Matrixbf[x][2]=1
            listc0.remove(i)
            listc1.remove(i)
        x=x+1
    elif i in set1.intersection(set2):
        d1=euclidean(Matrix2[i],m1)
        d2=euclidean(Matrix2[i],m2)
        d1=d1/(d1+d2)
        d2=d2/(d1+d2)
        d3=1
        Matrixb[x][0]=d1
        Matrixb[x][1]=d2
        Matrixb[x][2]=d3
        d=min(d1,d2,d3)
        if(d1==d):
            Matrixbf[x][0]=1
            Matrixbf[x][1]=0
            Matrixbf[x][2]=0
            listc1.remove(i)
        if(d2==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=1
            Matrixbf[x][2]=0
            listc0.remove(i)
        if(d3==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=0
            Matrixbf[x][2]=1
        x=x+1
    elif i in set2.intersection(set3):
        d2=euclidean(Matrix2[i],m2)
        d3=euclidean(Matrix2[i],m3)
        d2=d2/(d2+d3)
        d3=d3/(d2+d3)
        d1=1
        Matrixb[x][0]=d1
        Matrixb[x][1]=d2
        Matrixb[x][2]=d3
        d=min(d1,d2,d3)
        if(d1==d):
            Matrixbf[x][0]=1
            Matrixbf[x][1]=0
            Matrixbf[x][2]=0
        if(d2==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=1
            Matrixbf[x][2]=0
            listc2.remove(i)
        if(d3==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=0
            Matrixbf[x][2]=1
            listc1.remove(i)
        x=x+1
    elif i in set3.intersection(set1):
        d3=euclidean(Matrix2[i],m3)
        d1=euclidean(Matrix2[i],m1)
        d3=d1/(d1+d2)
        d1=d2/(d1+d2)
        d2=1
        Matrixb[x][0]=d1
        Matrixb[x][1]=d2
        Matrixb[x][2]=d3
        d=min(d1,d2,d3)
        if(d1==d):
            Matrixbf[x][0]=1
            Matrixbf[x][1]=0
            Matrixbf[x][2]=0
            listc2.remove(i)
        if(d2==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=1
            Matrixbf[x][2]=0
        if(d3==d):
            Matrixbf[x][0]=0
            Matrixbf[x][1]=0
            Matrixbf[x][2]=1
            listc0.remove(i)
        x=x+1


#print(Matrixb)
#print(Matrixbf)

#print("hard cluster k means k=3")
#print("first cluster")
#print(listc0)
#print("second cluster")
#print(listc1)
#print("third cluster")
#print(listc2)

#print(len(listc0))
#print(len(listc1))
#print(len(listc2))

listsi=[]
def silhoutte(l1,l2,l3):
    si=0
    for i in range(150):
        if i in l1:
            a=0
            b=0
            for j in l1:
                a=a+distance.euclidean(Matrix1[i],Matrix1[j])
            for k in l2:
                b=b+distance.euclidean(Matrix1[i],Matrix1[k])
            for l in l3:
                b=b+distance.euclidean(Matrix1[i],Matrix1[l])
            s=(b-a)/max(b,a)
        if i in l2:
            a=0
            b=0
            for j in l2:
                a=a+distance.euclidean(Matrix1[i],Matrix1[j])
            for k in l1:
                b=b+distance.euclidean(Matrix1[i],Matrix1[k])
            for l in l3:
                b=b+distance.euclidean(Matrix1[i],Matrix1[l])
            s=(b-a)/max(b,a)
        if i in l3:
            a=0
            b=0
            for j in l3:
                a=a+distance.euclidean(Matrix1[i],Matrix1[j])
            for k in l2:
                b=b+distance.euclidean(Matrix1[i],Matrix1[k])
            for l in l1:
                b=b+distance.euclidean(Matrix1[i],Matrix1[l])
            s=(b-a)/max(b,a)
        si=si+s
        listsi.insert(i,s)
    sav=si/150
    return sav
print(" for 1st clustering:")
sil=silhoutte(listc0,listc1,listc2)
print("silhoutte index:")
print(sil)
#print(listsi)

def intra(l):
    maxd=0
    for i in l:
        for j in l:
            d=distance.euclidean(Matrix1[i],Matrix1[j])
            if(d>maxd):
                maxd=d
    return maxd
def inter(l1,l2):
    maxd=0
    for i in l1:
        for j in l2:
            d=distance.euclidean(Matrix1[i],Matrix1[j])
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
print("dunns index:")
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

from random import seed
from random import randint
a=np.array(Matrix)
def rand(f):
        value = randint(0,f)
        return value

def mean_list(l):
    m=0.0
    for i in l:
        m=m+i
    return m/len(l)
        
mean1=rand(w)
mean2=rand(w)
mean3=rand(w)
#print(mean1)
#print(mean2)
#print(mean3)
lk1=[]
lk2=[]
lk3=[]
mk1=-1
mk2=-1
mk3=-1

while(mean1!=mk1 and mean2!=mk2 and mean3!=mk3):
    mk1,mk2,mk3=mean1,mean2,mean3
    lk1=[]
    lk2=[]
    lk3=[]
    lk1.append(mean1)
    lk2.append(mean2)
    lk3.append(mean3)
    for i in range(w):
        d1=distance.euclidean(Matrix1[mean1],Matrix1[i])
        d2=distance.euclidean(Matrix1[mean2],Matrix1[i])
        d3=distance.euclidean(Matrix1[mean3],Matrix1[i])
        d=min(d1,d2,d3)
        if(d==d1):
             lk1.append(i)
        if(d==d2):
             lk2.append(i)
        if(d==d3):
             lk3.append(i)
    mean1=mean_list(lk1)
    mean2=mean_list(lk2)
    mean3=mean_list(lk3)
    mean1=int(mean1)
    mean2=int(mean2)
    mean3=int(mean3)
    if(mean1!=mk1 and mean2!=mk2 and mean3!=mk3):
            del(lk1)
            del(lk2)
            del(lk3)

#print(lk1)
#print(lk2)
#print(lk3)
 
print("kmeans algo:")
sil=silhoutte(lk1,lk2,lk3)
print("silhoutte:")
print(sil)
#print(listsi)

dunn=dunns(lk1,lk2,lk3)
print("dunns index:")
print(dunn)


dbi=db(lk1,lk2,lk3)
print("db index:")
print(dbi)
listk=[]
listk5=[]
listkd=[]
import matplotlib.pyplot as plt
for i in range (w):
   listk=[]
   for j in range (w):
        d=distance.euclidean(Matrix1[i],Matrix1[j])
        listk.append(d)
   listk.sort()
   listk5.append(listk[3])
   d=(listk[0]+listk[1]+listk[2]+listk[3])/4
   listkd.append(d)
   del(listk)
listk5.sort()
listkd.sort()
#plt.plot(listk5,listkd) 

#plt.xlabel('sorted 5th nearest neibour dist') 
#plt.ylabel('5th nearest neibour dist') 
  
#plt.title('finding esp k=5') 
#plt.show()

eps,k=.6,4
def density_reach(l):
    for i in l:
       listdb=[]
       for j in range (w):
          d=distance.euclidean(Matrix1[i],Matrix1[j])
          listdb.append(d)
       for k in range (len(listd)):
          if(listd[k]<eps and k not in l):
              l.append(k)
       del(listdb)
    return l

listdb1=[]
listdb2=[]
listdb3=[]
listd=[]
count=0
for i in range (w):
  if(i not in listdb1 and i not in listdb2 and i not in listdb3):
    c=0
    listd=[]
    for j in range (w):
        d=distance.euclidean(Matrix1[i],Matrix1[j])
        listd.append(d)
    for k in listd:
        if(k<eps):
            c=c+1
    if(c>=k):
        count=count+1
        if(count==1):
           listdb1.append(i)
           listdb1=density_reach(listdb1)
        if(count==2):
           listdb2.append(i)
           listdb2=density_reach(listdb2)
        if(count==3):
           listdb3.append(i)
           listdb3=density_reach(listdb3)
    
#print(listdb1)
#print(listdb2)
#print(listdb3)

print("DBScan algo:")
sil=silhoutte(listdb1,listdb2,listdb3)
print("silhoutte index:")
print(sil)
#print(listsi)

dunn=dunns(listdb1,listdb2,listdb3)
print("dunns index:")
print(dunn)


dbi=db(listdb1,listdb2,listdb3)
print("db index:")
print(dbi)  

Matrix3 = [[0 for x in range(w)] for y in range(w)]
for i in range (w):
    for j in range (w):
              d=distance.euclidean(Matrix1[j],Matrix1[i])
              Matrix3[i][j]=d
