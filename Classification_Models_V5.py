#!/usr/bin/env python
# coding: utf-8

# ## Classification Models performance evaluation

# In[ ]:


# # if needed--- should be install first.
# pip install portion 
# pip install xgboost


# In[2]:


import warnings

warnings.filterwarnings('ignore')

import statistics

import keras
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import portion as P
import seaborn as sns
from keras import callbacks
from keras.callbacks import History
from keras.regularizers import l2

#import hvplot.pandas
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

# # Machine Learning models for classification task.
# 
# Now we've got our data split into training and test sets, it's time to build a machine learning model.
# 
# We'll train it (find the patterns) on the training set.
# 
# And we'll test it (use the patterns) on the test set.

# In[3]:


def Performance_Classification(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    num_folds=3,
    Accuracy_Score=True,
    AUC=True,
    Precision_Score=True,
    Robustness=False,
    Keras=False,
):
    """Calculate in-sample and out-of-sample performance score as well as the robustness checking score for classification model
    ----------------
    Input:
    model: the classification model
    X_train: training predictive variables (list of explanatory variables)
    X_test: tested predictive variables- for out off sample performance evaluation purpose
    y_train: training dependent variable
    y_test: tested dependent variable
    num_folds: number of folds for corss-validation, default set to 3
    --------------------
    Output:
    IRS_in: in sample performance score ranking between 0 and 1 (higher score indicates better performance)
    IRS_off: out off sample performance score ranking between 0 and 1 (higher score indicates better performance)
    Robutsness_score: robustness checking score also ranking between 0 and 1 (higher score indicates better performance)
    """
    IRS_in = []
    IRS_off = []

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    if y_pred_train.ndim > 1:
        y_pred_train = np.argmax(y_pred_train, axis=1)
    if y_pred_test.ndim > 1:
        y_pred_test = np.argmax(y_pred_test, axis=1)

    if Accuracy_Score:
        # accuracy score for training & testing sample...
        train_score = accuracy_score(y_train, y_pred_train)
        test_score = accuracy_score(y_test, y_pred_test)
        IRS_in.append(train_score)
        IRS_off.append(test_score)

    if AUC:
        # AUC score for training & testing sample
        #         y_pred_probs = model.predict_proba(X_train)[:, 1]
        #         fpr, tpr, thresholds = roc_curve(y_train, y_pred_probs)
        #         IRS_in_2=auc(fpr, tpr)
        #         y_pred_probs1 = model.predict_proba(X_test)[:, 1]
        #         fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_probs1)
        #         IRS_off_2=auc(fpr1, tpr1)
        #         IRS_in.append(IRS_in_2)
        #         IRS_off.append(IRS_off_2)
        IRS_in_2 = roc_auc_score(y_train, y_pred_train)
        IRS_off_2 = roc_auc_score(y_test, y_pred_test)
        IRS_in.append(IRS_in_2)
        IRS_off.append(IRS_off_2)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    if Precision_Score:
        # precision for training & testing sample...
        IRS_in_3 = precision_score(y_train, y_pred_train)
        IRS_off_3 = precision_score(y_test, y_pred_test)
        IRS_in.append(IRS_in_3)
        IRS_off.append(IRS_off_3)
    #     # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score

    #     # recall for training & testing sample...
    #     IRS_in_4=recall_score(y_train, model.predict(X_train))
    #     IRS_off_4=recall_score(y_test, model.predict(X_test))
    # https://www.w3schools.com/python/python_ml_confusion_matrix.asp#:~:text=Sensitivity%20(Recall),-Of%20all%20the&text=Sensitivity%20(sometimes%20called%20Recall)%20measures,been%20incorrectly%20predicted%20as%20negative).

    IRS_in = statistics.mean(IRS_in)
    IRS_off = statistics.mean(IRS_off)
    # robustness checking using only 5 iterations.
    # https://scikit-learn.org/stable/modules/cross_validation.html
    """ calculate the cross validation score for logistic regression with 3 iteration (cv=3)
    For robustness checking pupose.
    """

    if Robustness:
        inputs = np.concatenate((X_train, X_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)

        if Keras:
            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=num_folds, shuffle=True)

            history1 = History()
            acc_per_fold = []
            for train, test in kfold.split(inputs, targets):
                model
                model.fit(
                    inputs[train],
                    targets[train],
                    epochs=100,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[
                        history1,
                        keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            min_delta=0,
                            patience=10,
                            verbose=0,
                            mode="auto",
                        ),
                    ],
                )

                scores = model.evaluate(inputs[test], targets[test], verbose=0)
                acc_per_fold.append(scores[1])

            y_pred_train = model.predict(X_train)
            if y_pred_train.ndim > 1:
                y_pred_train = np.argmax(y_pred_train, axis=1)
            train_score = accuracy_score(y_train, y_pred_train)

            scores = [number - train_score for number in acc_per_fold]
            Robutsness_score = 1 - np.mean(np.abs(scores))

        else:
            scores = cross_val_score(model, inputs, targets, cv=num_folds)
            score_temp = np.subtract(scores, train_score)
            Robutsness_score = 1 - np.mean(np.abs(score_temp))
            # Robutsness_score=1-scores.std()

        return round(IRS_in, 4), round(IRS_off, 4), round(Robutsness_score, 4)

    return round(IRS_in, 4), round(IRS_off, 4)


# # Model 1: Logistic Regression

# In[5]:


# X = data.drop('target', axis=1) # we assume that RF is the predictive (target) variable
# y = data.target
# X_train, X_test, y_train, y_test=data_split(data)


# In[6]:


# lr_clf = LogisticRegression(solver='liblinear')
# lr_clf.fit(X_train, y_train)

# Performance_Classification(lr_clf, X_train, X_test, y_train, y_test, X, y)


# # Model 2: K-nearest neighbors

# In[7]:


# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_train)

# Performance_Classification(knn_clf, X_train, X_test, y_train, y_test, X, y)


# # Model 3: Support Vector machine

# In[8]:


# svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
# svm_clf.fit(X_train, y_train)

# Performance_Classification(svm_clf, X_train, X_test, y_train, y_test, X, y)


# # Model 4: Decision Tree Classifier

# In[9]:


# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_clf.fit(X_train, y_train)

# Performance_Classification(tree_clf, X_train, X_test, y_train, y_test, X, y)


# # Model 5: Random Forest

# In[10]:


# rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
# rf_clf.fit(X_train, y_train)

# Performance_Classification(rf_clf, X_train, X_test, y_train, y_test, X, y)


# # Model 6. XGBoost Classifer

# In[11]:


# xgb_clf = XGBClassifier(use_label_encoder=False)
# xgb_clf.fit(X_train, y_train)


# In[12]:


# Performance_Classification(xgb_clf, X_train, X_test, y_train, y_test, X, y)


# # Uncertainty quantification for classification models using Neural Network ensemble method.
# 
# 

# In[42]:


def compute_bj(start, end, step_size):
    ''' 
    First, discretize the observed confidence c into some number (K) 
    and return a list of K number of expected confidence each between 0 and 1.
    Then, compute confidence internvals associated with each confidence level.
    -------------------
    Input:
    start: the lower value of the expected confidence
    end: the upper value of the expected confidence
    step_size: the size of the discretization process
    ------------
    Output:
    discretized_interval: list of expected confidence levels.
    K: the number of expected confidence levels.
    bj_values: a collection of confidence intervals for each confidence level.
    '''
    discretized_interval = np.arange(start, end + step_size, step_size)
    K=len(discretized_interval)
    bj_values = [(cj - 1/(2* K), cj + 1/(2* K)) for cj in discretized_interval]
    
    return discretized_interval, K, bj_values


# In[44]:


def observed_Confidence(start, end, step_size, X_test, y_test, X_train, y_train,y_pred_probs,Bj_values):
    '''
    Calculate the observed confidence levels for each expected confidence level.
    -------------
    Input:
    start: the lower value of the expected confidence
    end: the upper value of the expected confidence
    step_size: the size of the discretization process
    X_test: tested dataset about explanatory variables
    y_test: tested dataset about the dependant variable
    -------------
    Output:
    c_j: a vector of size K that contents all the observed confidence levels.
    '''
#    Bj_values=compute_bj(start, end, step_size)[2]
    c_j = []
    for B_j in Bj_values:
        #interval = P.closed(*B_j)
        #interval="({}, {}]".format(B_j[0], B_j[1])
        
        # Calculate two components that will be used as numerator/denominator 
        # to compute the observed confidence level.
        lower_bound= B_j[0]
        upper_bound=B_j[1]
        count = 0
        counts = 0
        num_rows=X_test.shape[0]
        for j in range(0,num_rows):
            #row_x_arr = np.array([float(row_x)])
            #X_new = X_test.iloc[[j]]
            c=y_pred_probs[j,:]
            y_val=y_test.iloc[j]
            #f=c[y_val]
            f=c[1]
            if lower_bound<f<=upper_bound:
                # numerator of the observed confidence level
                count += 1
                # denominator of the observed confidence level
                counts += y_val * 1        
        
        if count!=0:
            c_j.append(counts/count)
        else:
            c_j.append(0)
    return c_j


# In[45]:


def weigth_function(X_test, B_j, X_train, y_train,y_pred_probs):
    '''
    Calculate the vector of weight associated with each observed confidence level.
    It called one function- Proba_Value- to have the average probability function
    ----------
    Input:
    X_test:tested dataset about explanatory variables
    B_j: a confidence interval
    ----------
    Output:
    w: the vector of weight
    '''
    lower_bound= B_j[0]
    upper_bound=B_j[1]
    #c=y_pred_probs
    w=0
    num_rows=X_test.shape[0]
    for j in range(num_rows):
        c=y_pred_probs[j,:]
        if lower_bound<c[1]<=upper_bound:
            w += 1
    return w


# In[ ]:


def Arg_function (X_train, X_test, y_train, y_test):
#     X_train, X_test, y_train, y_test = train_test_split(X_Ca, 
#                                                     ys, 
#                                                     test_size=0.3, random_state=42)
    lr_clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    knn_clf = KNeighborsClassifier().fit(X_train, y_train)
    tree_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    xgb_clf = XGBClassifier(use_label_encoder=False).fit(X_train, y_train)
    
    y_pred_probs_1 = lr_clf.predict_proba(X_test)
    y_pred_probs_2 = knn_clf.predict_proba(X_test)
    y_pred_probs_3 = tree_clf.predict_proba(X_test)
    y_pred_probs_4 = rf_clf.predict_proba(X_test)
    y_pred_probs_5 = xgb_clf.predict_proba(X_test)
    
    y_pred_probs=(y_pred_probs_1+y_pred_probs_2+y_pred_probs_3+y_pred_probs_4+y_pred_probs_5)/5
    
    return lr_clf, knn_clf,tree_clf, rf_clf, xgb_clf, y_pred_probs


# In[4]:


def ECE_value(X_train, X_test, y_train, y_test, start=0, end=1, step_size=0.1, random_state=42):
    '''
    Calculate the uncertainty quantification score ranking between 0 and 100%.
    It called 4 functions in the process
    ----------
    Input:
    start: the lower value of the expected confidence
    end: the upper value of the expected confidence
    step_size: the size of the discretization process
    X_test: tested dataset about explanatory variables
    y_test: tested dataset about the dependant variable
    -----------
    Output:
    IRS_uncertainty: Uncertainty quantification score.
    '''
    lr_clf, knn_clf,tree_clf, rf_clf, xgb_clf, y_pred_probs=Arg_function(X_train, X_test, y_train, y_test)
    c_exp,K,Bj_values=compute_bj(start, end, step_size)
    c_obs=observed_Confidence(start, end, step_size, X_test, y_test, X_train, y_train,y_pred_probs,Bj_values)

    w_j = []
    for B_j in Bj_values:
        w_j.append(weigth_function(X_test, B_j, X_train, y_train,y_pred_probs))
    N_m=sum(w_j)
    num_rows=X_test.shape[0]
    ECE=0
    for j in range(0,K):
        ECE +=w_j[j]*abs(c_exp[j]-c_obs[j])
    sample_size = len(X_test)
    ECE_1=ECE/sample_size
    IRS_uncertainty=round((1-ECE_1),4)
    return IRS_uncertainty   


# # Uncertainty quantification using Conformal Prediction

# In[ ]:


# MCR and SS

# Get class counts
def get_class_counts(y_test, n_classes):
    class_counts = []
    for i in range(n_classes):
        class_counts.append(np.sum(y_test == i))
    return class_counts

# Get coverage for each class
def get_coverage_by_class(prediction_sets, y_test, n_classes):
    coverage = []
    for i in range(n_classes):
        coverage.append(np.mean(prediction_sets[y_test == i, i]))
    return coverage

# Get average set size for each class
def get_average_set_size(prediction_sets, y_test, n_classes):
    average_set_size = []
    for i in range(n_classes):
        average_set_size.append(
            np.mean(np.sum(prediction_sets[y_test == i], axis=1)))
    return average_set_size     

# Get weighted coverage (weighted by class size)
def get_weighted_coverage(coverage, class_counts):
    total_counts = np.sum(class_counts)
    weighted_coverage = np.sum((coverage * class_counts) / total_counts)
    weighted_coverage = round(weighted_coverage, 4)
    return weighted_coverage

# Get weighted set_size (weighted by class size)
def get_weighted_set_size(set_size, class_counts):
    total_counts = np.sum(class_counts)
    weighted_set_size = np.sum((set_size * class_counts) / total_counts)
    weighted_set_size = round(weighted_set_size, 4)
    return weighted_set_size


# In[ ]:


def UQ_Conformal_Prediction(model, X_Cal, y_cal, X_test, y_test, class_labels=[0,1], n_classes=2, alpha=0.05):
    
    '''
    Input args:
        'model': model used for clasification (e.g., model=LogisticRegression(random_state=42).fit(X_train, y_train))
        'X_Cal': X calibration set
        'y_cal': y calibration set
        'X_test': X test set
        'y_test': y test set
        'class_labels': list contains class labels
        'n_classes': number of classes in the classfication model.
        'alpha': float between 0 to 1, determines the percentile threshold (i.e., 95th percentile when 𝛼 = 0.05)
        
    Output:
        'MCR'
    
    '''
    # Calculate conformal prediction threshold
    # Get predictions for calibration set
    if X_Cal.ndim == 2:
        y_pred_proba = model.predict_proba(X_Cal)
    if X_Cal.ndim == 3:
        y_pred_proba = model.predict(X_Cal)
    # y_pred_proba = model.predict_proba(X_Cal)
    
    # LAC
    si_scores = []
    # Loop through all calibration instances
    y_cal = y_cal.astype(int)
    for i, true_class in enumerate(y_cal):
        # Get predicted probability for observed/true class
        predicted_prob = y_pred_proba[i][true_class]
        si_scores.append(1 - predicted_prob) 

    # Convert to NumPy array
    si_scores = np.array(si_scores)    

    # Get threshold based on user-defined alpha 
    number_of_samples = len(X_Cal)
    qlevel = (1 - alpha) * ((number_of_samples + 1) / number_of_samples)
    threshold = np.percentile(si_scores, qlevel*100)
    
    # Get samples/classes from test set classified as positive
    if X_Cal.ndim == 2:
        prediction_sets = (1 - model.predict_proba(X_test) <= threshold)
    if X_Cal.ndim == 3:
        prediction_sets = (1 - model.predict(X_test) <= threshold)
    
    results = pd.DataFrame(index=class_labels)
    results['Class counts'] = get_class_counts(y_test, n_classes)
    results['Coverage'] = get_coverage_by_class(prediction_sets, y_test, n_classes)
    # results['Average set size'] = get_average_set_size(prediction_sets, y_test, n_classes)

    MCR = get_weighted_coverage(results['Coverage'], results['Class counts'])
    # SS = get_weighted_set_size(results['Average set size'], results['Class counts'])  
    
    return MCR


#  # To test uncertainty quantification...

# In[11]:


# start = 0
# end = 1
# step_size = 0.1
# #probability_function=lr_clf.predict_proba
# p=2
# X = data.drop('target', axis=1) # we assume that RF is the predictive (target) variable
# y = data.target
# #num_rows=X_test.shape[0]


# In[21]:


# lr_clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)
# knn_clf = KNeighborsClassifier().fit(X_train, y_train)
# svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0).fit(X_train, y_train)
# tree_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
# rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42).fit(X_train, y_train)
# xgb_clf = XGBClassifier(use_label_encoder=False).fit(X_train, y_train)


# In[47]:


# ECE_value(start, end, step_size, X_test, y_test, p, X_train, y_train)


# # References

# https://resilience-datafabric-dev.sbp.eyclienthub.com/gpt-foundry/

# #### Classification models
# 
#   https://www.kaggle.com/code/faressayah/practical-guide-to-6-classification-algorithms

# #### Uncertainty quantification nns
# 
# https://www.geeksforgeeks.org/ensemble-methods-in-python/

# In[ ]:




