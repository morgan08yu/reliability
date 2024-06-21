#!/usr/bin/env python
# coding: utf-8

# ## Data availability assessment
# 
# Focus on evaluating the data availability.

# In[2]:


import random
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple 


# ### Load testing data

# In[71]:


# df=pd.read_csv('F-F_Research_Data_Factors.CSV')

# Add np.nan to dataframe for illustration
# nanidx1 = df['Mkt-RF'].sample(frac=0.05).index
# nanidx2 = df['SMB'].sample(frac=0.2).index
# df.loc[nanidx1, 'Mkt-RF'] = np.NaN
# df.loc[nanidx2, 'SMB'] = np.NaN


# In[76]:


# df.head()


# ### Data availability - % of available data

# In[29]:


def Availability_IRS(num_var_needed, df, prct=0.1):
    '''Calculates the percentage of available data
    ----------
    Input: 
        'num_var_needed': float (positive number), total number of variables needed.
        'df': dataframe contains only variable(s) of interest
        'prct': float (between 0 and 1), missing rate threshods set by users to determine 
                if the variable should be calssified as available or not. (i.e.,prct = 0.1 
                suggests a variable with missing value over 10% should be considered as unavailable)
    ----------    
    Return: 
        'availability_rate': float (between 0 and 1)
      
    '''   
    if num_var_needed == 0:
        raise ValueError("num_var_needed should be greater than 0") 
    
    num_avail_var = len(df.columns)
    
    # If a variable contains missing value - check whether the missing rate is above acceptable threshold
    for col in df.columns[df.isna().any()]:
        missing_prct = (df[col].isna().sum())/len(df)
        if missing_prct > prct:
            num_avail_var = num_avail_var-1
        
    availability_rate = num_avail_var/num_var_needed
    
    if availability_rate > 1:
        availability_rate = 1
    
    IRS = round(availability_rate,4)
    
    return IRS 


# In[34]:


# def Availability_IRS(num_var_needed, df, prct=0.1):
#     '''Determines the individual realibility score (IRS) of completness using thresholds* below:
#     IRS=1 if Metric=100%, 
#     IRS=0.95 if 99%<Metric<100%; 
#     IRS= 0.9 if 95%<=Metrics<99%; 
#     IRS=0.8 if 90%<=Metric<95%; 
#     IRS=0.7 if 80%<=Metric<90%; 
#     IRS=0.6 if 70%<=Metric<80%
#     *Thresholds are provided by subject experts
#     ----------
#     Input: 
#         'num_var_needed': float (positive number), total number of variables needed.
#         'df': dataframe contains only variable(s) of interest
#         'prct': float (between 0 and 1), missing rate threshods set by users to determine 
#                 if the variable should be calssified as available or not. (i.e.,prct = 0.1 
#                 suggests a variable with over 10% missing value should be considered as unavailable)
#     ----------    
#     Return: 
#         'IRS': float (between 0 and 1)
#     '''  
#     availability_rate = availability(num_var_needed,df,prct)
    
#     if availability_rate == 1:
#         IRS = 1
#     if 0.99 < availability_rate < 1:
#         IRS = 0.95
#     if 0.95 <= availability_rate < 0.99:
#         IRS = 0.9
#     if 0.90 <= availability_rate < 0.95:
#         IRS = 0.8
#     if 0.80 <= availability_rate < 0.90:
#         IRS = 0.7    
#     if 0.70 <= availability_rate < 0.80:
#         IRS = 0.6   
#     elif availability_rate < 0.70:
#         IRS = 0
    
#     return IRS      


# In[74]:


# Availability_IRS(4,df,0.1)


# In[ ]:




