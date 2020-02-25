import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from pytorch_models import GRU_classifier,LSTM_classifier,GRU_regressor,LSTM_regressor
import torch 



def data_frames_to_train_pytorch(df_lst,look_back):
    
    #First will construct new features
    D = []

    for i,df in enumerate(df_lst):
        d = []
        
        for j in range(look_back,len(df)-1):
            avg = np.mean(df.iloc[:j,4])
            features = [df.iloc[j,4]-df.iloc[j-num,4] for num in range(1,look_back+1)]
            features.append(df.iloc[j,4]-avg)
            if df.iloc[j+1,4]/df.iloc[j,4] > 1:
                target = 0
            #elif df.iloc[j+1,4]/df.iloc[j,4] == 1:
            #    target = 1
            else:
                target = 1
            d.append((torch.Tensor(features),target))
            
        D.append(d)
    return D 



def data_frames_to_train_pytorchR(df_lst,look_back):
    
    #First will construct new features
    D = []
    At = []
    Ac = []
    for i,df in enumerate(df_lst):
        d = []
        at = []
        ac = []
        for j in range(look_back,len(df)-1):

            avg = np.mean(df.iloc[:j,4])
            features = [df.iloc[j,4]-df.iloc[j-num,4] for num in range(1,look_back+1)]
            features.append(df.iloc[j,4]-avg)
            target = df.iloc[j+1,4]-df.iloc[j,4] 

            d.append((torch.Tensor(features),target))
            at.append(df.iloc[j+1,4])
            ac.append(df.iloc[j,4])
        At.append(at)
        Ac.append(ac)
        D.append(d)
    return D,At,Ac 