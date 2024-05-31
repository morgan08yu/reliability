#!/usr/bin/env python
# coding: utf-8

# ## Structured data representativeness
# 
# Focus on categorical and numerical variables.

# In[112]:


import random # helpful link...https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
from scipy import stats # will use kstest (Kolmogorov-Smirnov Test) for numerical variables and chisquare for categorical variables
import numpy as np
import pandas as pd
import os
from statsmodels.stats.stattools import medcouple 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


# ## Dataset 
# The first thing is to download the training dataset as well as the reference/population dataset.

# ## A proposed function to compute the PSI index
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html 
#     
# https://github.com/mwburke/population-stability-index/blob/master/psi.py#L75

# In[114]:


def psi(expected_array, actual_array, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI for a single variable

    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into

    Returns:
       psi_value: calculated PSI value
    '''

    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

    def sub_psi(e_perc, a_perc):
        '''Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)

    psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

    return(psi_value)


# In[1]:


def variable_identifier(data,
                        cat_rules = [],
                        num_rules = [],
                        bin_rules = [],
                        date_rules = [],
                        binary =None):
    """
    This function identifies categorical,
    numerical and binary variables in a
    give data set.

    Returns: binary_col, cat_col, num_col, date_col

    Args:
            'cat_rules': a list of all the categorical identification strings
            'bin_rules': a list of all the binary identification strings
            'binary': binary variables are considered as a category by itself
            'numerical': binary variables are considered as a numerical varaible
            'categorical': binary variables are considered as categorical variable
    """
    binary_col = []
    cat_col = []
    num_col = []
    date_col = []
    date_col = []

    for var in data.columns:
        
        num_check = []
        cat_check = []
        bin_check = []
        date_check = []
        for rule in num_rules:
            num_check.append(rule in var)
        for rule in cat_rules:
            cat_check.append(rule in var)
        for rule in bin_rules:
            bin_check.append(rule in var)
        for rule in date_rules:
            date_check.append(rule in var)

        n_unique = data[var].nunique()
        sequence = np.sort(data[var].value_counts().index)
        if n_unique == 2:
            binary_col.append(var)
        elif sum(bin_check) > 0:
            binary_col.append(var)
        elif sum(date_check) > 0:
            date_col.append(var)
        elif data[var].dtype == np.dtype('object'):
            cat_col.append(var)
        elif sum(cat_check) > 0:
            cat_col.append(var)
        elif sum(num_check) > 0:
            num_col.append(var)
        elif all([x in [1,2,3,4] for x in np.diff(sequence)]):
            cat_col.append(var)
        else:
            num_col.append(var)
    if binary == 'binary':
        return binary_col, cat_col, num_col, date_col
    elif binary == 'numerical':
        num_col = num_col + binary_col
        return cat_col, num_col, date_col
    else:
        cat_col = cat_col + binary_col

        return cat_col, num_col, date_col


# In[4]:


def create_ref_df(df_train,df_test):
    numberV1 = df_train.shape[1]
    numberV2 = df_test.shape[1]
    ns=numberV1-numberV2
    if ns>0:
        # Find the missing columns in tested set
        missing_columns = [col for col in df_train.columns if col not in df_test.columns]
        # Add the missing columns to tested set with NaN values
        for col in missing_columns:
            #df_test[col] = df_test[col].apply(lambda x: np.random.choice(df_train[col].dropna()))
            df_test[col] = pd.Series(dtype=df_train[col].dtype)
        # Reorder the columns in tested set to match the order in training set
        df_test = df_test[df_train.columns]
        for col in missing_columns:
            if col in df_train.columns and col in df_test.columns:
                df_test[col] = df_test[col].apply(lambda x: np.random.choice(df_train[col].dropna()))
        
    elif ns<0:
        missing_columns = [col for col in df_test.columns if col not in df_train.columns]
        for col in missing_columns:
            #df_train[col] = df_train[col].apply(lambda x: np.random.choice(df_test[col].dropna()))
            df_train[col] = pd.Series(dtype=df_test[col].dtype)
        df_train = df_train[df_test.columns]
        for col in missing_columns:
            if col in df_test.columns and col in df_train.columns:
                df_train[col] = df_train[col].apply(lambda x: np.random.choice(df_test[col].dropna()))
    else:
        df_test = df_test[df_train.columns]
                 
    # importance of each dataset in the reference dataset...
    N_train=df_train.shape[0]
    N_test=df_test.shape[0]
    N_total=N_train+N_test
    tau_1=N_test/N_total
    if tau_1<0.40:
        N_test_2=int(0.40*N_total)
        diffs=N_test_2-N_test
        df_augmented = resample(df_test, replace=True, n_samples=diffs, random_state=42)
    df_test1=pd.concat([df_test, df_augmented])
    df_reference=pd.concat([df_train, df_test1])
    
    return df_reference


# ## Compute the individuel reliability associated with data representativeness component
# Two different metrics are used to evaluate the data representativeness.
# 
# And the two tests are performed for each individual variable we have in our database.

# In[2]:


def Data_representativeness_score( df_train, df_reference, significant=0.05):
    '''
    Calculate the data representativeness score using - PSI index
   
    ----------------
    Input:
    df_train: The training dataset that contents all the variables 
    df_reference: The reference dataset
    significant: The significant level of the test
    Number_V: Number of variable to be tested.
    -----------------
    Output:
    IRS: The data representativenss score using PSI.  
    '''
    categorical_columns = df_train.select_dtypes(include=['object', 'category']).columns
        # transform the categorical data into numerical data
    #Cat_training_processed = pd.DataFrame()
    le = LabelEncoder()
    for column in categorical_columns:
        df_train[column] = le.fit_transform(df_train[column])
        df_reference[column] = le.fit_transform(df_reference[column])

    
     # match the shape of reference data with training data
    diff = len(df_train) - len(df_reference)
    if diff > 0:
        df_augmented = resample(df_reference, replace=True, n_samples=abs(diff), random_state=42)
        df_reference = pd.concat([df_reference, df_augmented])

    elif diff < 0:
        # Reference_df = df_reference.sample(n=len(df_train))
        df_augmented = resample(df_train, replace=True, n_samples=abs(diff), random_state=42)
        df_train = pd.concat([df_train, df_augmented])

    count=0
    Number_Var=df_train.shape[1]
    for j in range(Number_Var):
        Varj=df_train.iloc[:, j]
        Refj=df_reference.iloc[:, j]
        Psi_value=psi(Varj, Refj, buckettype='bins', buckets=10, axis=0)
        if Psi_value<0.1:
            count=count+1
        elif 0.1<=Psi_value<0.25:
            count=count+0.5
    IRS=count/Number_Var
    return IRS


# In[ ]:





# In[ ]:





# # Numerical variables representativeness
# 
# Interpretation of kstest pvalue: A p-value that is less than or equal to your significance level indicates there is sufficient
#     evidence to conclude that the observed distribution is not the same as the expected distribution.
#     
# PSI Value Interpretation:
#               
# PSI < 0.1: Insignificant change—your model is stable! (no significant change in the two datasets)
# 
# 0.1 <= PSI < 0.25: Moderate change—consider some adjustments.
# 
# PSI >= 0.25: Significant change—your model is unstable and needs an update.  
#     
# 

# In[115]:


def NumericalV(Num_training, Num_ref, significant, Number_nuV):
    '''
    Calculate the numerical data representativeness score using two different statistical tests- PSI
    and Kolmogorov-Smirnov Test (KS Test)--- It return two different reliability score one for PSI and another for KS
    ----------------
    Input:
    Num_training: The training numerical variables
    Num_ref: The reference numerical variables
    significant: The significant level of the test
    Number_nuV: Number of numerical variable to be tested.
    -----------------
    Output:
    IRS1: The data representativeness score using kstest
    IRS2: The data representativenss score using PSI.  
    '''
    #count_1=0
    count_2=0
    for j in range(Number_nuV):
        Varj=Num_training.iloc[:, j]
        Refj=Num_ref.iloc[:, j]
        #stat, pvalue=stats.kstest(Varj, Refj)
        #if pvalue<significant:
        #count_1=count_1+1
        Psi_value=psi(Varj, Refj, buckettype='bins', buckets=10, axis=0)
        if Psi_value<0.1:
            count_2=count_2+1
        elif 0.1<=Psi_value<0.25:
            count_2=count_2+0.5
    #IRS1=1-(count_1/Number_nuV)
    IRS=count_2/Number_nuV
    return IRS


# # Categorical variables representativeness
# 
#  Chi-Square Test :A p-value that is less than or equal to your significance level indicates there is sufficient
#     evidence to conclude that the observed distribution is not the same as the expected distribution.

# In[116]:


def CategoricalV(Cat_training, Cat_ref, significant, Number_caV):
    '''
    Calculate the categorical data representativeness score using two different statistical tests- PSI
    and chisquare Test--- It return two different reliability score one for PSI and another for ch2-test
    ----------------
    Input:
    Cat_training: The training categorical variables
    Cat_ref: The reference categorical variables
    significant: The significant level of the test
    Number_caV: Number of categorical variable to be tested.
    -----------------
    Output:
    IRS1: The data representativeness score using chi2-test
    IRS2: The data representativenss score using PSI.  
    '''
    
    # transform the categorical data into numerical data
    Cat_training_processed = pd.DataFrame()
    le = LabelEncoder()
    for column in Cat_training.columns:
        Cat_training_processed[column] = le.fit_transform(Cat_training[column])

    Cat_ref_processed = pd.DataFrame()
    le = LabelEncoder()
    for column in Cat_ref.columns:
        Cat_ref_processed[column] = le.fit_transform(Cat_ref[column])
        
        
    #count_1=0
    count_2=0
    for j in range(Number_caV):
        Varj=Cat_training_processed.iloc[:, j]
        Refj=Cat_ref_processed.iloc[:, j]
        #stat, pvalue=stats.chisquare(Varj, Refj)
        #if pvalue<significant:
        #count_1=count_1+1
        Psi_value=psi(Varj, Refj, buckettype='bins', buckets=10, axis=0)
        if Psi_value<0.1:
            count_2=count_2+1
        elif 0.1<=Psi_value<0.25:
            count_2=count_2+0.5
    #IRS1=1-(count_1/Number_caV)
    IRS=count_2/Number_caV
    return IRS


# # Function to combine Categorical & Numerical Scores
#  
#     

# In[20]:


def Data_repr_1(Num_training, Num_ref, significant, Number_nuV):
    '''
    Calculate the final numerical data representativeness score when we only have numerical variabes
    in the model.It return the final data representativeness score.
    ----------------
    Input:
    Num_training: The training numerical variables
    Num_ref: The reference numerical variables
    significant: The significant level of the test
    Number_nuV: Number of numerical variable to be tested.
    -----------------
    Output:
    IRS: The final data representativeness score
    '''
    IRS=NumericalV(Num_training, Num_ref, significant, Number_nuV)
    #IRS=min(IRS_1,IRS_2)
    return IRS


# In[21]:


def Data_repr_2(Cat_training, Cat_ref, significant, Number_caV):
    '''
    Calculate the final data representativeness score  when only categorical variables are
    available in the model.It return the final data representativeness score.
    ----------------
    Input:
    Cat_training: The training categorical variables
    Cat_ref: The reference categorical variables
    significant: The significant level of the test
    Number_caV: Number of categorical variable to be tested.
    -----------------
    Output:
    IRS: The final data representativeness score
    '''
    IRS=CategoricalV(Cat_training, Cat_ref, significant, Number_caV)
    #IRS=min(IRS_1,IRS_2)
    return IRS


# In[22]:


def Data_repr_3(Num_training, Num_ref, significant, Number_nuV,Cat_training, Cat_ref, Number_caV):
    '''
    Calculate the final data representativeness score  when we have both categorical and numerical variables in the model.
    It return the final data representativeness score.
    ----------------
    Input:
    Num_training: The training numerical variables
    Num_ref: The reference numerical variables
    significant: The significant level of the test
    Number_nuV: Number of numerical variable to be tested
    Cat_training: The training categorical variables
    Cat_ref: The reference categorical variables
    Number_caV: Number of categorical variable to be tested.
    -----------------
    Output:
    IRS: The final data representativeness score
    '''
    a=NumericalV(Num_training, Num_ref, significant, Number_nuV)
    b=CategoricalV(Cat_training, Cat_ref, significant, Number_caV)
    IRS=(a*Number_nuV+b*Number_caV)/(Number_nuV+Number_caV)
    #IRS_2=(b*Number_nuV+d*Number_caV)/(Number_nuV+Number_caV)
    #IRS=min(IRS_1,IRS_2)
    return IRS


# # Compute the final data representativeness score.

# In[23]:


def Representativeness_score(Num_training=None, Num_ref=None, Number_nuV=None, 
                             Cat_training=None, Cat_ref=None, Number_caV=None, 
                             significant=0.05, numerical=False, categorical=False, both=False):
    '''
    Calculate the final data representativeness score depending on the type of data we have (numerical, categorical or both).
    It return the final data representativeness score.
    ----------------
    Input:
    Num_training: The training numerical variables
    Num_ref: The reference numerical variables
    significant: The significant level of the test
    Number_nuV: Number of numerical variable to be tested
    Cat_training: The training categorical variables
    Cat_ref: The reference categorical variables
    Number_caV: Number of categorical variable to be tested.
    -----------------
    Output:
    IRS: The final data representativeness score
    '''
    
    if numerical:
        IRS=Data_repr_1(Num_training, Num_ref, significant, Number_nuV)
    elif categorical:
        IRS=Data_repr_2(Cat_training, Cat_ref, significant, Number_caV)
    elif both:
        IRS=Data_repr_3(Num_training, Num_ref, significant, Number_nuV, Cat_training, Cat_ref, Number_caV)

    return round(IRS,4)


# In[1]:


#pip install streamlit_jupyter


# In[2]:


#pip install streamlit


# In[ ]:




