#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:13:13 2019

@author: dhriti
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import simps
from numpy import trapz
def f():
    x = [0,1] 
    y = [0,1] 
    plt.plot(x, y)
    return
def roc(x,y,z):
      plt.plot(x, y) 
      plt.xlabel('x - axis') 
      plt.ylabel('y - axis') 
      plt.title(z)
      f()
      plt.show()
      y = np.array(y)
      x = np.array(x)
      area = trapz(y, x)
      print("area =", area)
      return

x = [0,.024,1] 
y = [0,0.953,1]
z='roc curve iris decision tree classifier'
roc(x,y,z)

x = [0,.027,1] 
y = [0,0.946,1]
z='roc curve iris naive-bayes classifier'
roc(x,y,z)

x = [0,.460,1] 
y = [0,0.600,1]
z='roc curve wine decision tree classifier'
roc(x,y,z)

x = [0,.380,1] 
y = [0,0.800,1]
z='roc curve wine naive-bayes classifier'
roc(x,y,z)
