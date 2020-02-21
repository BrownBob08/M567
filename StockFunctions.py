# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:05:14 2020

@author: conno
"""

import numpy as np



#Takes a vector and returns its difference vector.
#An mx1 vector will result in (m-1)x1 difference vector.
#A vector [x1, x2, ..., xm] will result in a difference vector [x2-x1, x3-x2, ..., xm-x(m-1)] where xm is our most recent observation
def difference_vec(vec): 
    
    m,n=vec.shape
    diffvec=np.zeros((m-1,1))
    for i in range(m-1):
        diffvec[i] = vec[i+1]-vec[i]
    return diffvec


#Creates training set for KNN. Creates a pair of vectors (training data, 'target/s')
def create_KNN_training_set(vec, points_used, points_predicted):
    m,n = vec.shape
    trainingset = []
    
    for i in range(m-(points_used+points_predicted)):
        pair = (vec[i:i+points_used],vec[i+points_used:i+points_used+points_predicted])
        trainingset.append(pair)
        
    return trainingset       


#takes the training data which is a list of tuples (training vector, value) and finds the average of the 
#K nearest neighbors to vec[0] or our observed vector. vec is a tuple so vec[1] is our target.
#change this to make vec just a vector when done testing accuracy and implementing.
#the function uses the 2-norm to measure similarity
def KNN_average(k,training_data, vec):
    similarity_set=[]
    result=0
    m = len(training_data)
    for i in range(m):
        similarity=np.linalg.norm(vec[0]-training_data[i][0])
        similarity_set.append(similarity)
    sorted_list = np.argsort(similarity_set)
    for j in range(k):
        id=sorted_list[j]
        result+= (training_data[id][1])
    result=result/k
    print(result-vec[1])
    print()
    return result
    
    
     