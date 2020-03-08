#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[73]:


testdata=pd.read_csv(r'C:\Users\conno\Documents\Data_Projects\FB AWS data\fb2020-03-05.csv')


# In[74]:


traindata1=pd.read_csv(r'C:\Users\conno\Documents\Data_Projects\FB AWS data\fb2020-02-28.csv')
traindata2=pd.read_csv(r'C:\Users\conno\Documents\Data_Projects\FB AWS data\fb2020-03-04.csv')
traindata3=pd.read_csv(r'C:\Users\conno\Documents\Data_Projects\FB AWS data\fb2020-03-03.csv')
traindata4=pd.read_csv(r'C:\Users\conno\Documents\Data_Projects\FB AWS data\fb2020-03-02.csv')


# In[75]:


x_vals=testdata['lastSaleTime']
y_vals=testdata['lastSalePrice']

plt.plot(x_vals,y_vals)


# In[76]:


maxlist = [traindata1['lastSalePrice'].max(),traindata2['lastSalePrice'].max(),traindata3['lastSalePrice'].max(),traindata4['lastSalePrice'].max()]
minlist = [traindata1['lastSalePrice'].min(),traindata2['lastSalePrice'].min(),traindata3['lastSalePrice'].min(),traindata4['lastSalePrice'].min(),]


# In[87]:


new_max=np.min(maxlist)
new_min=np.min(minlist)


# In[88]:


new_max


# In[89]:


new_min


# In[90]:


middle=np.mean([new_max,new_min])


# In[91]:


plt.plot(x_vals,y_vals)
plt.axhline(y=new_max, color='r', linestyle='-')
plt.axhline(y=new_min, color='r', linestyle='-')
plt.axhline(y=middle, color='r', linestyle='--')


# In[ ]:




