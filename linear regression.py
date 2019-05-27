#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:16:42 2019

@author: dhriti
"""

from pandas import DataFrame
from sklearn import linear_model
import pandas as pd
import numpy as np
dataset = pd.read_csv('/home/dhriti/Downloads/iris1.csv')
x = dataset.iloc[:, [0,1,2,3]].values

X1= dataset.iloc[:, [0]].values[:135]
X2= dataset.iloc[:, [1]].values[:135]
X3= dataset.iloc[:, [2]].values[:135]
X4= dataset.iloc[:, [3]].values[:135]
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
#print(X1)

df = DataFrame(x,columns=['X1','X2','X3','X4'])


X = df[['X1','X2']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['X4']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
#print(regr.coef_[0])

# prediction with sklearn
#New_Interest_Rate = 2.75
#New_Unemployment_Rate = 5.3
x11=dataset.iloc[:, [0]].values[-15:]
x22=dataset.iloc[:,[1]].values[-15:]
#print(x22)
output=[]
for i in range(15):
  k=x11[i]
  q=x22[i]

  output.append(regr.coef_[0]*k+regr.coef_[1]*q+regr.intercept_)
print(output)
import matplotlib.pyplot  as plt
x=[i for i in range (0,5)]
plt.plot(x,x,color='blue')
plt.scatter(output,dataset.iloc[:, [3]].values[-15:],color='red')
plt.show()
#print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

'''import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model 
import pandas as pd 

df = pd.read_csv("/home/dhriti/Downloads/iris.csv")
df1=pd.read_csv("/home/dhriti/Downloads/iris1.csv")
#new_df=df.dropna(axis = 0, how ='any')

X = df.iloc[:, [0]].values
Y= df.iloc[:, [2]].values
#Y = df['C'] 
#X = df['A'] 
   
X=X.reshape(len(X),1) 
Y=Y.reshape(len(Y),1) 
   
# Split the data into training/testing sets 
X_train = X[:-30] 
X_test = X[-30:] 
   
# Split the targets into training/testing sets 
Y_train = Y[:-30] 
Y_test = Y[-30:] 
   
# Plot outputs 
plt.scatter(X_test, Y_test,  color='black') 
plt.title('Test Data') 
plt.xlabel('A') 
plt.ylabel('C') 
plt.xticks(()) 
plt.yticks(()) 
   
  
# Create linear regression object 
regr = linear_model.LinearRegression() 
   
# Train the model using the training sets 
regr.fit(X_train, Y_train) 
   
# Plot outputs 
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3) 
plt.show() 
#print(Y_test)
#print(Y_train)
#print(regr.predict(X_test))
Y1= df1.iloc[:, [2]].values
Y1_test = Y1[-30:]

for i in range(30):
     if Y_test[i]==0:
         print(Y1_test[i]+regr.predict(X_test[i])+(regr.predict(X_test[i])-regr.predict(Y1_test[i])))
print('$')

a=((Y1_test)-regr.predict(X_test))
b=regr.predict(X_test)

table={}
for j in range(30):
    table[j]=[Y1_test[j],b[j],a[j]]
print(table)   
 
'''









 


