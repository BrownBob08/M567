# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:28:38 2020

@author: conno
"""

import numpy as np
import pandas as pd
from StockFunctions import difference_vec
from StockFunctions import create_KNN_training_set
from StockFunctions import KNN_average
import matplotlib.pyplot as plt

testdata=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\fb2020-02-19.csv')
traindata=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\fb2020-02-18 (2).tsv',sep='|')

trainvec=traindata['lastSalePrice'].to_numpy()
trainvec=trainvec[:,None]
diffvec=difference_vec(trainvec)
points_used=20
points_predicted=1
k=10

trainingset = create_KNN_training_set(diffvec, points_used, points_predicted)

testvec=testdata['lastSalePrice'].to_numpy()
testvec=testvec[:,None]
diffvec2=difference_vec(testvec)

totalerrorlist=[]
forecastlist=[]
actualdiff=[]
lastobsvals=[]

for i in range(len(diffvec2)-(points_used+points_predicted)):
    vec=(diffvec2[i:i+points_used],diffvec2[i+points_used:i+points_used+points_predicted])
    lastobs=testvec[points_used-1+i]
    forecast = KNN_average(k,trainingset, vec)
    forecastlist.append(float(forecast)+lastobs)
    totalerrorlist.append(float(abs(vec[1]-forecast)))
    actualdiff.append(float(vec[1]))
    lastobsvals.append(lastobs)
    
xpoints=[x for x in range(len(forecastlist))]
actualvals=[float(y[1]) for y in trainingset]
testvec=[x for sublist in testvec for x in sublist]
plt.plot(range(len(lastobsvals)),lastobsvals)
plt.plot(range(len(forecastlist)),forecastlist,'r-')


