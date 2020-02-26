import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




'''

Contains Four models:

    LSTM_classifier

    LSTM_regressor

    GRU_classifier

    GRU_regressor


Best Results have come from the regressors. Its not clear
if the GRU or LSTM have better results



'''

class customDataset(Dataset):

    def __init__(self, x, transform=None):
        self.data = x
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class GRU_classifier(nn.Module):
    '''
    Gated recurreny unit that tries to classify whether the next
    stock price will be above, below,or the same as the current price.


    '''

    def __init__(self,input_size,h_size,out_size=2):

        super(GRU_classifier, self).__init__()

        #size of hidden state
        self.input_size = input_size
        self.h_size = h_size 
        #intialize hidden state. 
        self.hidden = torch.zeros(h_size)
        #define x matricies
        self.Wx_z = nn.Linear(input_size,h_size)
        self.Wx_r = nn.Linear(input_size,h_size)
        self.Wx_h = nn.Linear(input_size,h_size)
        #define h matricies note that we dont need another bias 
        self.Wh_z = nn.Linear(h_size,h_size,bias=False)
        self.Wh_r = nn.Linear(h_size,h_size,bias=False)
        self.Wh_h = nn.Linear(h_size,h_size,bias=False)
        #define y_hat matrix 
        self.Wy = nn.Linear(h_size,out_size)

    def forward(self,x):
        h = self.hidden 
        zt = torch.sigmoid(self.Wx_z(x)+self.Wh_z(h))
        rt = torch.sigmoid(self.Wx_r(x)+self.Wh_r(h))
        self.hidden = zt*h + (1-zt)*torch.tanh(self.Wx_h(x)+self.Wh_h(rt*h))
        y_hat = self.Wy(self.hidden)
        return y_hat 

    def train(self,data_lst,epochs = 10,batch_size=32):
        '''
            The Data is in the list becasue data from diffrent days
            need to be treated differently. Specifically the 
            hidden state from the previous day probably should not 
            be reused. 
        '''
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        loader_lst = [DataLoader(customDataset(data)) for data in data_lst]
        for epoch in range(epochs):
            total_loss = 0 
            np.random.shuffle(loader_lst)
            for data_loader in loader_lst:
                loss = 0
                i = 1
                self.hidden = torch.zeros(self.h_size)
                for d,t in data_loader:
                    if i%batch_size == 0:
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.data 
                        loss = 0
                    y_hat = self.forward(d)
                    loss += criterion(y_hat,t)
                    i+= 1
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            print('epoch {} loss {}'.format(epoch,total_loss))




class LSTM_classifier(nn.Module):
    '''
    LSTM that tries to classify whether the next
    stock price will be above, below,or the same as the current price.


    '''

    def __init__(self,input_size,h_size,out_size=2):

        super(LSTM_classifier, self).__init__()

        #size of hidden state
        self.input_size = input_size
        self.h_size = h_size 
        #intialize hidden state. 
        self.hidden = torch.zeros(h_size)
        self.cell = torch.zeros(h_size)
        #define x matricies
        self.Wx_f = nn.Linear(input_size,h_size)
        self.Wx_i = nn.Linear(input_size,h_size)
        self.Wx_o = nn.Linear(input_size,h_size)
        self.Wx_c = nn.Linear(input_size,h_size)
        #define h matricies note that we dont need another bias 
        self.Wh_f = nn.Linear(h_size,h_size,bias=False)
        self.Wh_i = nn.Linear(h_size,h_size,bias=False)
        self.Wh_o = nn.Linear(h_size,h_size,bias=False)
        self.Wh_c = nn.Linear(h_size,h_size,bias=False)
        #define y_hat matrix 
        self.Wy = nn.Linear(h_size,out_size)

    def forward(self,x):
        h = self.hidden 
        i = torch.sigmoid(self.Wx_i(x)+self.Wh_i(h))
        f = torch.sigmoid(self.Wx_f(x)+self.Wh_f(h))
        o = torch.sigmoid(self.Wx_o(x)+self.Wh_o(h))
        c_til = torch.tanh(self.Wx_c(x)+self.Wh_c(h))
        self.cell = f*self.cell + i*c_til 
        self.hidden = o*torch.tanh(self.cell)
        y_hat = self.Wy(self.hidden)
        return y_hat 

    def train(self,data_lst,epochs = 10,batch_size=32):
        '''
            The Data is in the list becasue data from diffrent days
            need to be treated differently. Specifically the 
            hidden state from the previous day probably should not 
            be reused. 
        '''
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        loader_lst = [DataLoader(customDataset(data)) for data in data_lst]
        for epoch in range(epochs):
            total_loss = 0 
            np.random.shuffle(loader_lst)
            for data_loader in loader_lst:
                loss = 0
                i = 1
                self.hidden = torch.zeros(self.h_size)
                self.cell = torch.zeros(self.h_size)
                for d,t in data_loader:
                    if i%batch_size == 0:
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.data 
                        loss = 0
                    y_hat = self.forward(d)
                    loss += criterion(y_hat,t)
                    i+= 1
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            print('epoch {} loss {}'.format(epoch,total_loss))




class GRU_regressor(nn.Module):
    '''
    Gated recurreny unit that tries to classify whether the next
    stock price will be above, below,or the same as the current price.


    '''

    def __init__(self,input_size,h_size,out_size=1):

        super(GRU_regressor, self).__init__()

        #size of hidden state
        self.input_size = input_size
        self.h_size = h_size 
        #intialize hidden state. 
        self.hidden = torch.zeros(h_size)
        #define x matricies
        self.Wx_z = nn.Linear(input_size,h_size)
        self.Wx_r = nn.Linear(input_size,h_size)
        self.Wx_h = nn.Linear(input_size,h_size)
        #define h matricies note that we dont need another bias 
        self.Wh_z = nn.Linear(h_size,h_size,bias=False)
        self.Wh_r = nn.Linear(h_size,h_size,bias=False)
        self.Wh_h = nn.Linear(h_size,h_size,bias=False)
        #define y_hat matrix 
        self.Wy = nn.Linear(h_size,out_size)

    def forward(self,x):
        h = self.hidden 
        zt = torch.sigmoid(self.Wx_z(x)+self.Wh_z(h))
        rt = torch.sigmoid(self.Wx_r(x)+self.Wh_r(h))
        self.hidden = zt*h + (1-zt)*torch.tanh(self.Wx_h(x)+self.Wh_h(rt*h))
        y_hat = self.Wy(self.hidden).reshape(1)
        return y_hat 

    def train(self,data_lst,epochs = 10,batch_size=32,validation_set = None):
        '''
            The Data is in the list becasue data from diffrent days
            need to be treated differently. Specifically the 
            hidden state from the previous day probably should not 
            be reused. 
        '''
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters())
        loader_lst = [DataLoader(customDataset(data)) for data in data_lst]
        if validation_set != None:
            test_loader = DataLoader(customDataset(validation_set))
            self.hidden = torch.zeros(self.h_size)
            val_loss = 0
            for d,t in test_loader:
                y_hat = self.forward(d)
                val_loss += criterion(y_hat,t).data
            print('initial validation loss ',val_loss)
        for epoch in range(epochs):
            total_loss = 0 
            np.random.shuffle(loader_lst)
            for data_loader in loader_lst:
                loss = 0
                i = 1
                self.hidden = torch.zeros(self.h_size)
                for d,t in data_loader:
                    if i%batch_size == 0:
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.data 
                        loss = 0
                    y_hat = self.forward(d)
                    loss += criterion(y_hat,t)
                    i+= 1
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            if validation_set != None:
                self.hidden = torch.zeros(self.h_size)
                val_loss = 0
                for d,t in test_loader:
                    y_hat = self.forward(d)
                    val_loss += criterion(y_hat,t).data
                print('epoch {} train loss {} validation loss {}'.format(epoch,total_loss,val_loss))

            else:
                print('epoch {} train loss {}'.format(epoch,total_loss))


class LSTM_regressor(nn.Module):
    '''
    Gated recurrent unit that tries predict the difference between the next price and the current
    price


    '''

    def __init__(self,input_size,h_size,out_size=1):

        super(LSTM_regressor, self).__init__()

        #size of hidden state
        self.input_size = input_size
        self.h_size = h_size 
        #intialize hidden state. 
        self.hidden = torch.zeros(h_size)
        self.cell = torch.zeros(h_size)
        #define x matricies
        self.Wx_f = nn.Linear(input_size,h_size)
        self.Wx_i = nn.Linear(input_size,h_size)
        self.Wx_o = nn.Linear(input_size,h_size)
        self.Wx_c = nn.Linear(input_size,h_size)
        #define h matricies note that we dont need another bias 
        self.Wh_f = nn.Linear(h_size,h_size,bias=False)
        self.Wh_i = nn.Linear(h_size,h_size,bias=False)
        self.Wh_o = nn.Linear(h_size,h_size,bias=False)
        self.Wh_c = nn.Linear(h_size,h_size,bias=False)
        #define y_hat matrix 
        self.Wy = nn.Linear(h_size,out_size)

    def forward(self,x):
        h = self.hidden 
        i = torch.sigmoid(self.Wx_i(x)+self.Wh_i(h))
        f = torch.sigmoid(self.Wx_f(x)+self.Wh_f(h))
        o = torch.sigmoid(self.Wx_o(x)+self.Wh_o(h))
        c_til = torch.tanh(self.Wx_c(x)+self.Wh_c(h))
        self.cell = f*self.cell + i*c_til 
        self.hidden = o*torch.tanh(self.cell)
        y_hat = self.Wy(self.hidden).reshape(1)
        return y_hat 

    def train(self,data_lst,epochs = 10,batch_size=32,validation_set = None):
        '''
            The Data is in the list becasue data from diffrent days
            need to be treated differently. Specifically the 
            hidden state from the previous day probably should not 
            be reused. 
        '''
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters())
        loader_lst = [DataLoader(customDataset(data)) for data in data_lst]
        if validation_set != None:
            test_loader = DataLoader(customDataset(validation_set))
            self.hidden = torch.zeros(self.h_size)
            self.cell = torch.zeros(self.h_size)
            val_loss = 0
            for d,t in test_loader:
                y_hat = self.forward(d)
                val_loss += criterion(y_hat,t).data
            print('initial validation loss ',val_loss)
        for epoch in range(epochs):
            total_loss = 0 
            np.random.shuffle(loader_lst)
            for data_loader in loader_lst:
                loss = 0
                i = 1
                self.hidden = torch.zeros(self.h_size)
                self.cell = torch.zeros(self.h_size)
                for d,t in data_loader:
                    if i%batch_size == 0:
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.data 
                        loss = 0
                    y_hat = self.forward(d)
                    loss += criterion(y_hat,t)
                    i+= 1
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            if validation_set != None:
                self.hidden = torch.zeros(self.h_size)
                self.cell = torch.zeros(self.h_size)
                val_loss = 0
                for d,t in test_loader:
                    y_hat = self.forward(d)
                    val_loss += criterion(y_hat,t).data
                print('epoch {} train loss {} validation loss {}'.format(epoch,total_loss,val_loss))

            else:
                print('epoch {} train loss {}'.format(epoch,total_loss))