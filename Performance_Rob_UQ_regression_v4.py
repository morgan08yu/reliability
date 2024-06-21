#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[3]:


import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import scipy
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import keras
import keras.backend as k
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, LSTM, Activation, Masking, Dropout
from keras.callbacks import History
from keras import callbacks


# In[ ]:





# In[3]:


def Performance_Regression(model, X_train, X_test, y_train, y_test,
                               MAPE=True, RAE=True, MAE=True,
                               NRMSE=False, Robustness=False, R_2=False, Keras=False, num_folds=3):
    '''Calculate in-sample and out-of-sample performance score as well as the robustness checking score for classification model
    ----------------
    Input:
    model: the classification model 
    X_train: training predictive variables (list of explanatory variables)
    X_test: tested predictive variables- for out off sample performance evaluation purpose
    y_train: training dependent variable 
    y_test: tested dependent variable
    --------------------
    Output:
    IRS_in: in sample performance score ranking between 0 and 1 (higher score indicates better performance)
    IRS_off: out off sample performance score ranking between 0 and 1 (higher score indicates better performance)
    Robutsness_score: robustness checking score also ranking between 0 and 1 (higher score indicates better performance)
    '''
    IRS_in = []
    IRS_off = []
    
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    #if y_pred_train.ndim >1:
        #y_pred_train = np.argmax(y_pred_train,axis=1)
    #if y_pred_test.ndim >1:
        #y_pred_test = np.argmax(y_pred_test,axis=1)
        
    if MAPE:
        #MAPE mean absolute percentage error for training & testing sample...
        IRS_mape_is=1-mape(y_train,y_pred_train)
        #/np.mean(y_train)
        IRS_mape_oos=1-mape(y_test,y_pred_test)
        #/np.mean(y_test) 
        IRS_in.append(IRS_mape_is)
        IRS_off.append(IRS_mape_oos)
    
    if RAE:
    #RAE relative absolute error for training & testing sample
        RAE_is=mae(y_train,y_pred_train)/np.mean(abs(y_train-np.mean(y_train)))
        RAE_oos=mae(y_test,y_pred_test)/np.mean(abs(y_test-np.mean(y_test)))
        IRS_rae_is=1-RAE_is/np.mean(y_train)
        IRS_rae_oos=1-RAE_oos/np.mean(y_test)        
        IRS_in.append(IRS_rae_is)
        IRS_off.append(IRS_rae_oos)
    
    if MAE:
    # Mean Absolute Error for training & testing sample...
        IRS_mae_is=1-mae(y_train,y_pred_train)/np.mean(y_train)
        IRS_mae_oos=1-mae(y_test,y_pred_test)/np.mean(y_test)
        IRS_in.append(IRS_mae_is)
        IRS_off.append(IRS_mae_oos)
    
    if NRMSE:  
    # Normalized root mean squared error for training & testing sample...
        IRS_nrmse_is=1-((mse(y_train,y_pred_train))**(1/2))/np.mean(y_train)
        IRS_nrmse_oos=1-((mse(y_test,y_pred_test))**(1/2))/np.mean(y_test)
        IRS_in.append(IRS_nrmse_is)
        IRS_off.append(IRS_nrmse_oos)        
    if R_2:
    # Coefficient of determination
        R2_is=r2_score(y_train,y_pred_train)
        #1-(np.sum(np.square(y_train-y_pred_train)))/(np.sum(np.square(y_train-np.mean(y_train))))
        R2_oos=r2_score(y_test,y_pred_test)
        #1-(np.sum(np.square(y_test-y_pred_test)))/(np.sum(np.square(y_test-np.mean(y_test))))
        if R2_is<=0:
            IRS_r2_is=0
        else:
            IRS_r2_is=R2_is
        if R2_oos<=0:
            IRS_r2_oos=0
        else:
            IRS_r2_oos=R2_oos
        IRS_in.append(IRS_r2_is)
        IRS_off.append(IRS_r2_oos)
    IRS_in=statistics.mean(IRS_in)
    IRS_off=statistics.mean(IRS_off)
    #robustness checking using only 5 iterations.
    # https://scikit-learn.org/stable/modules/cross_validation.html
    ''' calculate the cross validation score for logistic regression with 3 iteration (cv=3)
    For robustness checking pupose.
    '''
    #if UQ:

    if Robustness:
        inputs = np.concatenate((X_train, X_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)

        if Keras:
            # Define the K-fold Cross Validator
            #num_folds=5
            kfold = KFold(n_splits=num_folds, shuffle=True)

            history1 = History()
            mse_per_fold = []
            for train, test in kfold.split(inputs, targets):
                model
                model.fit(inputs[train], targets[train], epochs=100, validation_split=0.1, verbose=1,
                      callbacks = [history1,
                                  keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')])

                scores = model.evaluate(inputs[test], targets[test], verbose=0)
                mse_per_fold.append(scores[1])

            y_pred_train=model.predict(X_train)
            if y_pred_train.ndim >1:
                y_pred_train = np.argmax(y_pred_train,axis=1)
            train_score = mse(y_train, y_pred_train) 

            scores = [number - train_score for number in mse_per_fold] 
            sc=pd.Series(mse_per_fold)
            Robutsness_score=1-sc.mad()/sc.mean()
            #Robutsness_score=1-np.mean(np.abs(scores))
        
        else:
            scores = cross_val_score(model, inputs, targets, cv=num_folds)
            score_temp = np.subtract(scores, train_score)
            Robutsness_score=1-np.mean(np.abs(score_temp))
            #Robutsness_score=1-scores.std()

        
        return round(IRS_in,4), round(IRS_off,4), round(Robutsness_score,4)
        
    
    return round(IRS_in,4), round(IRS_off,4)


# ## Aggregating Regression IRS 

# ## Robustness
# Cross validation, adversarial robustness ?: To see all possible scoring function to be used in cross_val_score do: sorted(sklearn.metrics.SCORERS.keys())

# In[ ]:





# In[2]:


def robust_keras_irs(input_train, input_test, target_train, target_test, sequence_length, nb_features, num_folds):
    # Merge inputs and targets
    inputs = np.concatenate((input_train, input_test), axis=0)
    targets = np.concatenate((target_train, target_test), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    # Define per-fold score containers <-- these are new
    mse_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(inputs, targets):
        # Define the model architecture
        history = History()
        model = Sequential()
        model.add(LSTM(
         units=100,
         return_sequences=True,
         input_shape=(sequence_length, nb_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(
          units=100,
          return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='relu'))
        model.add(Activation("relu"))
        # Compile the model
        model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])
        
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # Fit data to model
        model.fit(inputs[train], targets[train], epochs=100, batch_size=32, validation_split=0.1, verbose=1,
          callbacks = [history, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')])
        
        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        mse_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        
        # Increase fold number
        fold_no = fold_no + 1
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(mse_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {mse_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    sc=pd.Series(mse_per_fold)
    IRS=1-sc.mad()/sc.mean()
    return IRS


# In[ ]:





# In[ ]:


#robust_keras_irs(input_train, input_test, target_train, target_test, sequence_length, nb_features, num_folds)


# In[ ]:





# In[ ]:


def Rob_Regression_Score(X_train,y_train,n_split):
    '''
    Aggregate all the robustness individual score to provide the final regression model performance score
    ----------
    Input: 
        X_train: set of independent variables in the training sample.
        y_train: dependent variable in the training sample 
        n_split: number of cross validation samples (i.e., n_split=5 for 5 cv sample. 4 will be used as training and 1 left off for testing)
    ----------    
    Return: 
        'rob_regression_score': float (between 0 and 1)
    '''
    cv_r2 = robust_cv_r2(X_train,y_train,n_split)
    cv_mape = robust_cv_mape(X_train,y_train,n_split)
    rob_regression_score = min(1-cv_r2, 1-cv_mape)
    
    return rob_regression_score


# ## Uncertainty quantification

# 1. Specify predictive distribution: $y=X\theta+\epsilon$

# 2. Estimation using GPR,BNN,GMM,NNS: mean $\hat{\mu}=X\hat{\theta}$, variance: $\hat{\sigma}^2(X,\theta)$

# ### GPR 
# https://www.geeksforgeeks.org/gaussian-process-regression-gpr/

# In[4]:


'''
# Define the kernel (RBF kernel)
    kernel = 1.0 * RBF(length_scale=1.0)
    # Create a Gaussian Process Regressor with the defined kernel
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # Fit the Gaussian Process model to the training data
    gp.fit(X_train, y_train)
    # Make predictions on the test data
    mu, sigma = gp.predict(X_test, return_std=True)
    exp_CIs = np.linspace(1e-10, 1-1e-10, K) #expected CIs
    pred_CIs=[]                                 #predicted CIs
    for exp_CI in sorted(exp_CIs):
        intervals = scipy.stats.norm.interval(exp_CI, loc=mu, scale=sigma)
        lower_bd = intervals[0]
        upper_bd = intervals[1]
        pred_CIs.append(np.sum((y_test > lower_bd) & (y_test < upper_bd))/len(y_test))
    
    ECE = np.mean(np.abs(exp_CIs - pred_CIs))
'''


# In[71]:


def reg_GPR(X_train,X_test,y_train):
    """
    Input: 
    X_train, X_test: set of independent variables.
    y_train: dependent variable
    t_size: percentage of the data used for testing. Testing sample size. (i.e., t_size=0.10) 
    
    Output:
    mu: predicted mean
    sigma: predicted standard deviation
    y_test: dependent variable in the testing sample
    """
    # Define the kernel (RBF kernel)
    kernel = 1.0 * RBF(length_scale=1.0)
    # Create a Gaussian Process Regressor with the defined kernel
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # Fit the Gaussian Process model to the training data
    gp.fit(X_train, y_train)
    # Make predictions on the test data
    mu, sigma = gp.predict(X_test, return_std=True)
    #y_mean, y_cov = gp.predict(X_test, return_cov=True)
    return mu,sigma


# In[72]:


def reg_GPR_ECE(X_train,X_test,y_train,y_test,K):
    """
    Input: 
    X_train, X_test: set of independent variables.
    y_train, y_test: dependent variable 
    K: Number of confidence level chosen between 0 and 1
    
    Output:
    exp_CIs - expected confidence
    pred_CIs - predicted confidence
    ECE - Expected Calibrated Error
    """
    mu,sigma=reg_GPR(X_train,X_test,y_train)
    #c = np.arange(0, step_size, step_size)
    exp_CIs = np.linspace(1e-10, 1-1e-10, K) #expected CIs
    pred_CIs=[]                                 #predicted CIs
    for exp_CI in sorted(exp_CIs):
        intervals = scipy.stats.norm.interval(exp_CI, loc=mu, scale=sigma)
        lower_bd = intervals[0]
        upper_bd = intervals[1]
        pred_CIs.append(np.sum((y_test > lower_bd) & (y_test < upper_bd))/len(y_test))
    
    ECE = np.mean(np.abs(exp_CIs - pred_CIs))
    IRS=1-ECE
    return IRS  # converting to percentages


# ## Conformal prediction 
# Assessing uncertainty using conformal prediction based on https://arxiv.org/pdf/2107.07511.pdf

# In[ ]:


# import torch
# from torch import nn
import keras.backend as K
import numpy as np


# In[ ]:


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)


# In[ ]:


# nb_features = x_train.shape[2]
# nb_out = 1
# history = History()

# model = Sequential()
# model.add(LSTM(
#          units=100,
#          return_sequences=True,
#          input_shape=(sequence_length, nb_features)))
# model.add(Dropout(0.2))
# model.add(LSTM(
#           units=100,
#           return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=1, activation='relu'))
# model.add(Activation("relu"))
# model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])

# model.summary()


# # In[ ]:


# model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1,
#           callbacks = [history, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')])


# # In[ ]:


# model.predict_prob(x_test)


# # In[ ]:


# # Use reg_GPR to output mean and variance of the predictive distribution: Y_test | X_test
# loss = nn.GaussianNLLLoss()
# input = torch.randn(5, 2, requires_grad=True)
# target = torch.randn(5, 2)
# var = torch.ones(5, 2, requires_grad=True)  # heteroscedastic
# output = loss(input, target, var)
# output.backward()

# # scores
# scores=np.abs(y_true-y_pred)
# # quantiles

# # prediction sets. In the case of regression, prediction sets are a continuous-valued interval.





# # Get scores
# cal_scores = np.abs(cal_pred-cal_labels)/cal_U
# # Get the score quantile
# qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
# # Deploy (output=lower and upper adjusted quantiles)
# prediction_sets = [val_pred - val_U*qhat, val_pred + val_U*qhat ]

