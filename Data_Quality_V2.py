

import random
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple 
from sklearn.preprocessing import LabelEncoder


# ### Load data set

# In[3]:


#df=pd.read_csv('F-F_Research_Data_Factors.CSV')


# In[ ]:





# ### Data quality - Coherence

# In[5]:


def median_rule(vals, coef=1.5, inner_fences=False, plot=False, log_scale=True, **plot_kws):
    median = np.median(vals)
    iqr = stats.iqr(vals)
    lf = median - coef * iqr
    uf = median + coef * iqr
    if inner_fences:  # return inner fences
        uf = vals[vals <= uf].max()
        lf = vals[vals >= lf].min()
    # print('Upper {}fence: {} ({}%)'.\
    #    format('Inner ' if inner_fences else 'Outer ', uf, stats.percentileofscore(vals, uf)))
    if plot:  # always plot using log-scale
        lvals = vals
        luf = uf
        if not log_scale:
            lvals = np.log(vals)
            luf = np.log(uf)
        plot_kde(lvals, luf, True, **plot_kws)
    return lf, uf


# In[6]:


def adjboxStats(vals, coef=1.5, a=-4, b=3, inner_fences=False):
    '''     
    Adjusted Tukey Boxplot: Finds the upper and lower boundary for potential outliers 
    with skewness taking into account.
    
    Reference:
    Hubert, M.; Vandervieren, E. (2008). “An adjusted boxplot for skewed distribution”. 
    Computational Statistics and Data Analysis. 52 (12): 5186–5201.
    '''
    # print(stats.describe(vals))
    mc = medcouple(vals)
    q1, q3 = stats.mstats.mquantiles(vals, prob=[0.25, 0.75])
    iqr = stats.iqr(vals)
    if mc < 0:
        a_tmp = a
        a = -1 * b
        b = -1 * a_tmp
    lf = q1 - coef * iqr * np.exp(a * mc)  # -3.5, 4
    uf = q3 + coef * iqr * np.exp(b * mc)
    if inner_fences:
        uf = vals[vals <= uf].max()
        lf = vals[vals >= lf].min()
    # print('Upper {}fence: {} ({}%)'.\
    #   format('Inner ' if inner_fences else 'Outer ', uf, stats.percentileofscore(vals, uf)))
    return lf, uf


# In[7]:


def percent_outliers(var):
    '''Calculates the percentage of outliers in one variable
    ----------
    Input: 
        'var': array or series of the variable of interest (e.g., df['Price'])
    ----------    
    Return: 
        'outliers_rate': float (between 0 and 1)
      
    '''
    # Using adjusted tukey boxplot
    lf,uf = adjboxStats(var)
    outliers_rate = ((var>uf)|(var<lf)).sum()/len(var)
    
    return outliers_rate


# In[1]:


def Coherence_IRS(df):
    '''The individual realibility score (IRS) of coherence is derived by 1 - outliers_rate
    ----------
    Input: 
        'df': dataframe contains only variable(s) of interest 
    ----------    
    Return: 
        'IRS': float (between 0 and 1)
    '''  
    # remove nan
    df = df.dropna()
    
    # transform the categorical data into numerical data
    categorical_columns = df.select_dtypes(include=['category']).columns
    le = LabelEncoder()
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])

    # keep numerical columns
    df = df.select_dtypes(include=['number'])
    # Loop through all variables/columns in dataframe 'df'
    # Calculate outliers_rate using percent_outliers function for each variable
    outliers_rate_array = []
    for column in df:
        var = df[column]
        outliers_rate = percent_outliers(var)
        outliers_rate_array.append(outliers_rate)
    # Select the maximum outliers_rate to be more conservative
    outliers_rate = np.mean(outliers_rate_array)
    
    IRS = 1-outliers_rate

    
    return round(IRS,4)


# ### Data quality - Completeness

# In[38]:


def Completeness_IRS(data):
    '''
    Calculates the percentage of missing value in one variable or across variables in dataframe
    to derive the individual realibility score (IRS) of completness
    ----------
    Input: 
        'data': takes following format
            (1) dataframe contains only variable(s) of interest; 
            (2) list or array of variable(s) of interest; 
            (3) series of one variable of interest;
    ----------    
    Return: 
        'IRS': float (between 0 and 1)
    '''
    # when the input is a datarame
    if isinstance(data, pd.DataFrame):
        missing_value_rate = (data.isnull().sum()/len(data)).mean()
    
    # when the input is a list or an array
    if isinstance(data,list) or isinstance(data, np.ndarray): 
        data_2 = np.array(data)
        if len(data_2.shape)>1:
            data_2 = pd.DataFrame(data_2)
            missing_value_rate = (data_2.isnull().sum()/len(data_2)).mean()
        elif len(data_2.shape)==1:
            missing_value_rate = np.count_nonzero(np.isnan(data))/len(data)
            
    # when the input is a series
    if isinstance(data, pd.Series): 
        missing_value_rate = np.count_nonzero(np.isnan(data))/len(data)
    
    IRS = 1-missing_value_rate
    
    return round(IRS,4)


# ### Data quality - Uniqueness

# In[32]:


def Uniqueness_IRS(df):
    '''
    Calculates the percentage of duplicated rows in dataframe 'df' to derive the 
    individual realibility score (IRS) of uniqueness
    ----------
    Input: 
        'df': dataframe contains only variable(s) of interest; 
    ----------    
    Return: 
        'IRS': float (between 0 and 1)
    '''
    duplicate_rate = df.duplicated(keep=False).sum()/len(df)
    
    IRS = 1-duplicate_rate
    
    return round(IRS,4)


# ## Aggregate final data quality score

# In[34]:


def Data_Quality_Score(df,df_num):
    '''
    Aggregate all the data quality individual score to provide the final data quality score
    ----------
    Input: 
        'df': dataframe contains only variable(s) of interest;
        'df_num': dataframe contains only numerical variable(s) of interest for coherence checking
    ----------    
    Return: 
        'data_quality_score': float (between 0 and 1)
    '''
    uniqueness_irs = Uniqueness_IRS(df)
    completeness_irs = Completeness_IRS(df)
    coherence_irs = Coherence_IRS(df_num) 
    data_quality_score = min(uniqueness_irs, completeness_irs, coherence_irs)
    
    return data_quality_score


# In[ ]:


# coherence_irs only for numerical var
# diff weight for each part or diff aggregation method


# In[35]:


#Data_Quality(df)


# In[ ]:




