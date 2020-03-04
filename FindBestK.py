# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:35:00 2020

@author: conno
"""

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

traindata=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\fb2020-02-19.csv')
traindata2=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\fb2020-02-18 (2).tsv', delimiter='|')
traindata3=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\FB AWS data\\fb2020-02-20.csv')
traindata3=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\FB AWS data\\fb2020-02-21.csv')
traindata4=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\FB AWS data\\fb2020-02-24.csv')
traindata5=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\FB AWS data\\fb2020-02-25.csv')
traindata6=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\FB AWS data\\fb2020-02-26.csv')
testdata=pd.read_csv('C:\\Users\\conno\\Documents\\Data_Projects\\FB AWS data\\fb2020-02-27.csv')
    
trainvec=traindata['lastSalePrice'].to_numpy()
trainvec=trainvec[:,None]
diffvec_train=difference_vec(trainvec)

volume=testdata['volume'].to_numpy()
volume=volume[:,None]

testvec=testdata['lastSalePrice'].to_numpy()
testvec=testvec[:,None]
diffvec_test=difference_vec(testvec)


trainvec2=traindata2['lastSalePrice'].to_numpy()
trainvec2=trainvec2[:,None]
diffvec2_train=difference_vec(trainvec2)
diffvec_train=diffvec_train+diffvec2_train



trainvec3=traindata3['lastSalePrice'].to_numpy()
trainvec3=trainvec3[:,None]
diffvec3_train=difference_vec(trainvec3)
diffvec_train=diffvec_train+diffvec3_train

trainvec4=traindata4['lastSalePrice'].to_numpy()
trainvec4=trainvec4[:,None]
diffvec4_train=difference_vec(trainvec4)
diffvec_train=diffvec_train+diffvec4_train

trainvec5=traindata5['lastSalePrice'].to_numpy()
trainvec5=trainvec5[:,None]
diffvec5_train=difference_vec(trainvec5)
diffvec_train=diffvec_train+diffvec5_train

trainvec6=traindata6['lastSalePrice'].to_numpy()
trainvec6=trainvec6[:,None]
diffvec6_train=difference_vec(trainvec6)
diffvec_train=diffvec_train+diffvec6_train


trainvec=np.concatenate((trainvec,trainvec2,trainvec3,trainvec4,trainvec5,trainvec6))
errorlist=[]
dir_accuracy_list=[]
timestepsahead=3 
k=10
points_used=15
targetlist=[]
forecastdifflist=[]
forecastlist=[]
for j in range(len(testvec)-points_used-timestepsahead):
    indiv_test=testvec[j:j+points_used]
    indiv_testdiff=difference_vec(indiv_test)
    target=testvec[j+points_used+timestepsahead]#-indiv_test[-1]
    targetlist.append(float(target))
    similarityset=[]
    val_list=[]

    for i in range(len(trainvec)-points_used-timestepsahead):
        indiv_train = trainvec[i:i+points_used]
        actualdiff=trainvec[i+points_used+timestepsahead]-indiv_train[-1]
        val_list.append(actualdiff)
        indiv_traindiff = difference_vec(indiv_train)
        similarity = np.linalg.norm(indiv_testdiff-indiv_traindiff)
        similarityset.append(similarity)
    sortedlist=np.argsort(similarityset)
    forecastdiff=0
    for n in range(k):
        index=sortedlist[n]
        forecastdiff+=val_list[index]
    forecastdiff=forecastdiff/k
    forecastdifflist.append(float(forecastdiff))
    forecastlist.append(float(forecastdiff)+float(indiv_test[-1]))


        
plt.plot(range(len(targetlist)),targetlist)  
plt.plot(range(len(forecastlist)), forecastlist,'r')   
          
errorarray=abs(np.array(forecastlist)-np.array(targetlist))
error=sum(errorarray)/len(errorarray)

print(sum(errorarray)/len(errorarray))

wrong=0
for i in range(len(forecastdifflist)):
    if forecastdifflist[i]*targetlist[i]<0:
        wrong+=1
            
    direction_accuracy=1-wrong/len(forecastdifflist)
    errorlist.append(error)
    dir_accuracy_list.append(direction_accuracy)
    

