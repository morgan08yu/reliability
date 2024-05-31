# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:56:12 2024

@author: YR998EN
"""


import pickle
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# other useful package used by python.

import random
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.utils import resample
# transform categorical variables into numerical...
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
import lightgbm as lgbm
from typing import Optional, Dict, Union, Generator, Tuple
from tensorflow.keras.losses import sparse_categorical_crossentropy


# python functions used....


# import functions pre-defined in other ipynb files 
import import_ipynb
# make sure this file and those fuction ipynb files are in the same folder, otherwise specifies the path of function files

import Data_Quality_V2
import Data_Availability
import Data_Representativeness_V2
import Classification_Models_V5
# import Conformal_prediction_UQ_multiclass_classification
import Final_Aggregation

# if 'x_train' not in dir():
#     from DataModel_Aircraft_Baseline import * 
# else:
#     pass

# if 'x_train_2' not in dir():
#     from DataModel_Aircraft import * 
# else:
#     pass




#st.set_page_config(page_title="Multipage App", page_icon=())

st.title("Welcome to Reliability Index Demo")

st.sidebar.image("ernst-young-ey-logo-1536x1282.png")
st.sidebar.markdown("***")

model_selector:Optional[str] = st.sidebar.selectbox('Choose the model type to work on:', ('Classification','Regression','Clustering','Gen AI'), 
                                                    index=None, key='view_button1')
    
view_button: Optional[str] = st.sidebar.selectbox('Choose the modular to work on:', ('Upload File','Data Assessment', 'Model Performance Assessment',
                                                                                     'Ongoing Monitoring','Qualitative Assessment of Reliability',
                                                                                     'Business Relevance'), 
                                                  index=None, key='view_button2')
# if "score_vector" not in st.session_state:
#     st.session_state.score_vector = []
if "QS" not in st.session_state:
    st.session_state.QS = []    
if "Business" not in st.session_state:
    st.session_state.Business = []  
# if "Reliability" not in st.session_state:
#     st.session_state.Reliability = []  
if "weight_vector" not in st.session_state:
    st.session_state.weight_vector = [] 
    
if "Data_Availability" not in st.session_state:
    st.session_state.Data_Availability = []    
if "Data_Quality" not in st.session_state:
    st.session_state.Data_Quality = [] 
if "Data_Representativeness" not in st.session_state:
    st.session_state.Data_Representativeness = [] 
if "IRS_In" not in st.session_state:
    st.session_state.IRS_In = [] 
if "IRS_Off" not in st.session_state:
    st.session_state.IRS_Off = [] 
if "Robutsness_Score" not in st.session_state:
    st.session_state.Robutsness_Score = [] 
if "Uncertainty_Score" not in st.session_state:
    st.session_state.Uncertainty_Score = [] 


if "training_df" not in st.session_state:
    st.session_state.training_df = []    
if "testing_df" not in st.session_state:
    st.session_state.testing_df = [] 
if "ref_df" not in st.session_state:
    st.session_state.ref_df = []
if "target_var" not in st.session_state:   
    st.session_state.target_var = []
if "pretrained_model" not in st.session_state:   
    st.session_state.pretrained_model = []
if "x_train" not in st.session_state:   
    st.session_state.x_train = []
if "x_test" not in st.session_state:   
    st.session_state.x_test = []
if "y_train" not in st.session_state:   
    st.session_state.y_train = []
if "y_test" not in st.session_state:   
    st.session_state.y_test = []    
if "num_var" not in st.session_state: 
    st.session_state.num_var = []            
# ------------------------------------------------------------------------------
def fig_individual_score(score,dimension):
        
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['Reliability Index'],x=[50],name='Low',orientation='h',marker=dict(color='rgba(201, 17, 17, 0.8)'),width=0.2))
    fig.add_trace(go.Bar(y=['Reliability Index'],x=[30],name='Medium',orientation='h',marker=dict(color='orange'),width=0.2))
    fig.add_trace(go.Bar(y=['Reliability Index'],x=[20],name='High',orientation='h',marker=dict(color='rgba(17, 201, 84, 0.8)'),width=0.2))       
    fig.update_layout(autosize=False,width=550,height=250,xaxis=dict(showgrid=False,showline=False,showticklabels=False,zeroline=False),
        yaxis=dict(showgrid=False,showline=False,showticklabels=True,zeroline=False),barmode='stack',showlegend=True, 
        paper_bgcolor= "rgba(0, 0, 0, 0)",plot_bgcolor= "rgba(0, 0, 0, 0)",)
    fig.add_vline(x=score, line_width=2, line_dash="dash", line_color="rgba(161, 159, 155, 0.8)",
                  annotation_text=dimension+str(score)+"%", annotation_position="top")
    return fig

def qualitative_questionnaire():
    Q1 = ["Do you ensure data availability?", 
                 "Do you test the dataset for representativeness?",
                 "Do you check the data quality?",
                 "Do you apply a data quality issue mitigation strategy?",
                 "Aptness of model choice with business use and target data?",
                 "Do you test your model for over/under-fitting identification and remediation?",
                 "Do you perform an in-sample performance evaluation?",
                 "Do you perform an out-of-sample performance evaluation?",
                 "Robustness checking?",
                 "Benchmarking of your model?",
                 "Sensitivity analysis?",
                 "Do you perform uncertainty quantification for your model?",
                 "Do you regularly check the production data quality?",
                 "Do you regularly test the production data for data drift issues?",
                 "Do you regularly monitor the model performance drift?",
                 "Do you regularly monitor the model output stability?",
                 "Do you regularly update your model benchmarking?",
                 "Do you regularly collect users’ feedback & suggestions for future improvement of your model?",
                 "Is an action plan defined for the model retrain/recalibration/review?",
                 "Is there a fall-back mechanism in case the AI system is not performing as expected during production?"]
    
    Q2 = ["Do you do some intra family benchmarking?", 
                 "Do you do a traditional benchmarking?",
                 "Do you fine-tune your model in domain specific small dataset?",
                 "Do you test the performance of the model on a validation dataset to ensure it meets the desired level of accuracy?",
                 "Do you implement uncertainty quantification techniques to ensure model reliability and support informed decision-making?",
                 "Do you regularly check the production data quality?",
                 "Do you regularly test the production data for data drift issues?",
                 "Do you regularly monitor the model performance drift?",
                 "Do you regularly monitor the model output stability?",
                 "Do you regularly update your model benchmarking?",
                 "Do you regularly collect user’s feedback and suggestions for future improvement your model?",
                 "Is an action plan defined for the model retrain/recalibration/review?",
                 "Is there a fall-back mechanism in case the AI system is not performing as expected during production?"]
    
    
    # Create a dictionary to store user input
    user_input = {}
    
    
    response=st.selectbox("Is your model intended for prediction/classification purpose? ", ["Yes", "No"], index=None, placeholder="Please select...")
    if response=="Yes":
        # Iterate through the questions and create a selectbox for each
        for question in Q1:
            user_input[question] = st.selectbox(f"{question} ", ["Yes", "No", "NA"], key=question, index=None, placeholder="Please select...")
        # Count the occurrences of Yes, No
        count_yes = sum(1 for value in user_input.values() if value == "Yes")
        count_no = sum(1 for value in user_input.values() if value == "No")
        if count_yes+count_no!=0:
            qs = count_yes/(count_yes+count_no)
            st.write(f" The qualitative score is {round(100 * qs, 2)}%")
            st.session_state.QS = qs
    elif response=="No":
        response=st.radio("Are you using a pre-trained Gen AI? ", ('Yes', 'No'))
        if response=="Yes":
            # Iterate through the questions and create a selectbox for each
            for question in Q2:
                user_input[question] = st.selectbox(f"{question} ", ["Yes", "No", "NA"], key=question, index=None, placeholder="Please select...")
            # Count the occurrences of Yes, No
            count_yes = sum(1 for value in user_input.values() if value == "Yes")
            count_no = sum(1 for value in user_input.values() if value == "No")
            if count_yes+count_no!=0:
                qs = count_yes/(count_yes+count_no)
                st.write(f" The qualitative score is {round(100 * qs, 2)}%")
                st.session_state.QS = qs
        else:
            # Iterate through the questions and create a selectbox for each
            for question in Q1:
                user_input[question] = st.selectbox(f"{question}: Choose Yes/N0/NA", ["Yes", "No", "NA"], key=question)
            # Count the occurrences of Yes, No
            count_yes = sum(1 for value in user_input.values() if value == "Yes")
            count_no = sum(1 for value in user_input.values() if value == "No")
            if count_yes+count_no !=0:
                qs = count_yes/(count_yes+count_no)
                st.write(f" The qualitative score is {round(100 * qs, 2)}%")
                st.session_state.QS = qs

def business_impact():
    # Define questions
    questions = ["Q1. To what extent do the model’s output and its associated errors impact regulatory compliance?", 
                 "Q2. To what extent do the model’s output and its errors impact financial, or significant business decisions?",
                 "Q3. Is the AI in a public-facing role, potentially influencing public opinion or behaviors?", 
                 "Q4. To what extent the model output and its associated errors impact the organization’s reputation?"]
    
    # Create a dictionary to store user input
    user_input = {}
    
    # Iterate through the questions and create a selectbox for each
    for question in questions:
        user_input[question] = st.selectbox(f"{question} ", ["H", "M", "L", "NA"], key=question, index=None, placeholder="Please select...")
    
    # Count the occurrences of H, M, and L
    count_h = sum(1 for value in user_input.values() if value == "H")
    count_m = sum(1 for value in user_input.values() if value == "M")
    count_l = sum(1 for value in user_input.values() if value == "L")
    
    
    #response=st.st.selectbox("Is your model intended for prediction/classification purpose? ", ["Yes", "No"], index=None, placeholder="Please select...")
    
    response=st.selectbox("Is the AI use cases involved in life-critical or safety-critical operations?", ["Yes", "No"], index=None, placeholder="Please select...")
    
    if response=="Yes" or count_h !=0:
        st.write(f"The Business Impact is High (H)")
        st.session_state.Business="H"
    elif count_m !=0:
        st.write(f"The Business Impact is Medium (M)")
        st.session_state.Business="M"
    elif count_l !=0:
        st.write(f"The Business Impact is Low (L)")
        st.session_state.Business="L"
    else:
        st.write(f"The Business Impact is Not Applicable (NA)")
        st.session_state.Business="NA"

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
    

# --------------------------------------------------------------------------------------
   

# ---------------------------------------------------------------------------------------------
if model_selector == 'Classification':
    
        # # set up default file
        # # exec(open("Data_Model_Aircraft.py").read())
        # from DataModel_Aircraft import * 
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Upload File':
        st.subheader("Upload the data file")
        uploaded_data1 = st.file_uploader(label=":page_facing_up: Upload the training data file", type=["csv"])
        if uploaded_data1 is not None:
            training_df= pd.read_csv(uploaded_data1)
            st.session_state.training_df = training_df
            
        uploaded_data2 = st.file_uploader(label=":page_facing_up: Upload the testing data file", type=["csv"])
        if uploaded_data2 is not None:
            testing_df = pd.read_csv(uploaded_data2)
            st.session_state.testing_df = testing_df
            
        st.subheader("Upload the model file")
        uploaded_model = st.file_uploader(label=":page_facing_up: Upload the model file", type = ['pkl'])
        if uploaded_model is not None:
            # st.write(uploaded_model)
            # pretrained_model = pickle.load(open(uploaded_model.name, 'rb'))
            pretrained_model = pickle.load(uploaded_model)
            st.session_state.pretrained_model = pretrained_model
        
        st.subheader("Upload the AHP weight matrix file")
        Weight_AHP_data = st.file_uploader(label=":page_facing_up: Upload the AHP weight matrix file", type=["csv"],
                                           help='The AHP Matrix is constructed through pairwise comparisons, where weights are assigned to each comparison using the following scale: 1 (equal importance), 3 (somewhat more important), 5 (definitely more important), 7 (much more important), and 9 (very much more important).')
        if Weight_AHP_data is not None:
            Weight_AHP = pd.read_csv(Weight_AHP_data, index_col=0)
            weight_AHP = Weight_AHP.to_numpy()
            st.session_state.weight_vector = Final_Aggregation.Agg_weight(weight_AHP)
        
        st.markdown("***")
        if (uploaded_data1 is not None) & (uploaded_data2 is not None):
            st.subheader("Please provide more information about the data")
            if len(training_df.columns) != len(testing_df.columns):
                if len(training_df.columns) > len(testing_df.columns):
                    var_list = training_df.columns
                elif len(training_df.columns) < len(testing_df.columns):
                    var_list = testing_df.columns
            elif len(training_df.columns) == len(testing_df.columns):
                var_list = training_df.columns
                
            target_var = st.selectbox("Which column in your data represents the target variable? ", var_list, index=None, 
                                      help="The target variable is the main variable that you are interested in analyzing or predicting.",
                                      placeholder='Please select...') 
            if target_var is not None:
                st.session_state.target_var = target_var
                
            response = st.selectbox("Do you have a reference dataset that can be used to test the representativeness of your data? ", ('Yes','No'), 
                                    index=None, help="The reference dataset represents the true patterns and distributions you expect.",
                                    placeholder='Please select...') 
            if response == 'Yes':
                uploaded_data3 = st.file_uploader(label=":page_facing_up: Upload the reference data file", type=["csv"])
                if uploaded_data3 is not None:
                    ref_df = pd.read_csv(uploaded_data3)
                    st.session_state.ref_df = ref_df
                    
            elif response == 'No':
                ref_df = create_ref_df(training_df,testing_df)
                st.session_state.ref_df = ref_df
                st.markdown('A reference dataset has been created by sampling the training and testing data to reflect the patterns of complete dataset. This helps simulate the real-world scenarios that the model is expected to encounter.')
        
        if (uploaded_model is not None) & (uploaded_data1 is not None) & (uploaded_data2 is not None)  :
            response = st.selectbox("Are there any additional transformations required for your data before fitting it into the model?", ('Yes','No'), index=None, 
                                    help="For example, converting 2D data to 3D to better fit the model's requirements.",
                                    placeholder='Please select...') 
            if response =='Yes':
                trans_data = st.file_uploader(label=":page_facing_up: Upload the transformed data", type=["pkl"],
                                              help="This PKL file should include the following elements in this order: (1) transformed training data for explanatory variables, (2) transformed testing data for explanatory variables, (3) transformed training data for target variables, and (4) transformed testing data for target variables.")
                # st.write('This file should be in a .pkl format and should include the following:')
                # st.write(' - Transformed training data for explanatory variables.')
                # st.write(' - Transformed testing data for explanatory variables.')
                # st.write(' - Transformed training data for target variables.')
                # st.write(' - Transformed testing data for target variables.')
                if trans_data is not None:
                    data = []
                    with open('transformed_data.pkl', 'rb') as f:
                        while True:
                            try:
                                data.append(pickle.load(f))
                            except EOFError:
                                break
                    st.session_state.x_train = data[0]
                    st.session_state.x_test = data[1]
                    st.session_state.y_train = data[2]
                    st.session_state.y_test = data[3]
                
            elif response == 'No':
                st.session_state.x_train = st.session_state.training_df.drop([st.session_state.target_var], axis = 1)
                st.session_state.x_test = st.session_state.testing_df.drop([st.session_state.target_var], axis = 1)
                st.session_state.y_train = st.session_state.training_df[st.session_state.target_var]
                st.session_state.y_test = st.session_state.testing_df[st.session_state.target_var]
                

                
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Data Assessment':
        #section_options ='Data Assessment'
        st.subheader("Data availability assessment")        
        response1 = st.selectbox("Do you ensure the data availability? ", ('Yes', 'No'), index=None, placeholder="Please select...") 
        if response1 == 'No':
            st.write("Data availability assessment is essential to ensure AI reliability. Please perform necessary data availability assessment.")      
        if response1 == 'Yes':
            num_var = st.number_input("How many variables will you need for the analysis and model training, including both explanatory (input) and target (output) variables?", 
                                     min_value=1, value=None, 
                                     help='Explanatory variables are factors that might influence or predict your outcomes. Target variables are the outcomes you want to predict.',
                                     placeholder="Type a number...")
            if num_var is not None:
                st.session_state.number = num_var
                if st.session_state.Data_Availability == [] :
                    Availability_IRS = Data_Availability.Availability_IRS(num_var_needed=num_var, df = st.session_state.training_df, prct=0.4)
                    #st.write(f"The Data Availability Index is {round(100*Availability_IRS, 2)}%")
                    fig = fig_individual_score(round(Availability_IRS*100,2), 'Data Avalability score is ')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.Data_Availability = Availability_IRS
                else:
                    fig = fig_individual_score(round(st.session_state.Data_Availability*100,2), 'Data Avalability score is ')
                    st.plotly_chart(fig, use_container_width=True)
                
            
    # ---------------------------------------------------------------------------------------------
        Data_Quality_options2 = ['1', '2','3']
        Data_Quality_options3 = ['Data Coherence', 'Data Completeness', 'Data Uniqueness']
        
        st.subheader("Data quality assessment")
        response2 = st.selectbox("Do you assess data quality?", ('Yes', 'No'), index=None, placeholder="Please select...")
        if response2 == 'No':
            st.write("Data quality assessment is essential to ensure AI reliability. Please perform necessary data quality assessment.") 
        if response2 == 'Yes':
            response3 = st.selectbox("How many dimensions have you considered in data quality assessment?", Data_Quality_options2, 
                                     index=None, placeholder="Please select...")
            if response3 is not None:
                response3 = int(response3)
                if response3 == 1:
                    response4 = st.selectbox("Select the dimension that have been considered in data quality assessment", Data_Quality_options3, 
                                             index=None, placeholder="Please select...")
                    Data_Quality_temp = []
                    if response4 == 'Data Coherence':
                        Data_Quality_temp= Data_Quality_V2.Coherence_IRS(st.session_state.training_df)
                        #st.write(f"The Data Quality score is {round(100*Data_Quality, 2)}%")
                    if response4 == 'Data Completeness':
                        Data_Quality_temp = Data_Quality_V2.Completeness_IRS(st.session_state.training_df)
                        #st.write(f"The Data Quality score is {round(100*Data_Quality, 2)}%")
                    if response4 == 'Data Uniqueness':
                        Data_Quality_temp = Data_Quality_V2.Uniqueness_IRS(st.session_state.training_df)
                        #st.write(f"The Data Quality score is {round(100*Data_Quality, 2)}%") 
                    
                    if Data_Quality_temp!= []:
                        fig = fig_individual_score(round(Data_Quality_temp*100,2), response4+' score is ')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.Data_Quality = Data_Quality_temp
                    
                if response3 > 1:
                    response5 = st.multiselect("Select the dimension that have been considered in data quality assessment", Data_Quality_options3, 
                                               max_selections=response3, placeholder="Please select...")       
                    Data_Quality2 = []
                    if 'Data Coherence' in response5:
                        Data_Quality_Coherence = Data_Quality_V2.Coherence_IRS(st.session_state.training_df)
                        #st.write(f"The Data Coherence score is {round(100*Data_Quality_Coherence, 2)}%")
                        Data_Quality2.append(Data_Quality_Coherence)
                        fig = fig_individual_score(round(Data_Quality_Coherence*100,2), 'Data Coherence score is ')
                        st.plotly_chart(fig, use_container_width=True)  
                        
                    if 'Data Completeness' in response5:
                        Data_Quality_Completeness = Data_Quality_V2.Completeness_IRS(st.session_state.training_df)
                        #st.write(f"The Data Completeness score is {round(100*Data_Quality_Completeness, 2)}%")
                        Data_Quality2.append(Data_Quality_Completeness)
                        fig = fig_individual_score(round(Data_Quality_Completeness*100,2), 'Data Completeness score is ')
                        st.plotly_chart(fig, use_container_width=True) 
                        
                    if 'Data Uniqueness' in response5:
                        Data_Quality_Uniqueness = Data_Quality_V2.Uniqueness_IRS(st.session_state.training_df)
                        #st.write(f"The Data Uniqueness score is {round(100*Data_Quality_Uniqueness, 2)}%")
                        Data_Quality2.append(Data_Quality_Uniqueness)
                        fig = fig_individual_score(round(Data_Quality_Uniqueness*100,2), 'Data Uniqueness score is ')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    if Data_Quality2 != []:
                        Data_Quality_temp = min(Data_Quality2)
                        #st.write(f"The Data Quality score is {round(100*Data_Quality, 2)}%") 
                        fig = fig_individual_score(round(Data_Quality_temp*100,2), 'Data Quality score is ')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.Data_Quality = Data_Quality_temp
                        
        
    # ---------------------------------------------------------------------------------------------
        st.subheader("Data representativeness assessment") 
        response6 = st.selectbox("Do you assess data representativeness?", ('Yes', 'No'), index=None, placeholder="Please select...") 

        if response6 == 'No':
            st.write("Data representativeness assessment is essential to ensure AI reliability. Please perform necessary data representativeness assessment.")      
        if response6 == 'Yes':
            if st.session_state.Data_Representativeness == []:
                Representativeness_IRS=Data_Representativeness_V2.Data_representativeness_score(st.session_state.training_df, 
                                                                                                st.session_state.ref_df, 
                                                                                                significant=0.05)
                #st.write(f"The Data Representativeness Index is {round(100*Representativeness_IRS, 2)}%")
                
                fig = fig_individual_score(round(Representativeness_IRS*100,2), 'Data Representativeness score is ')
                st.plotly_chart(fig, use_container_width=True)    
                
                st.session_state.Data_Representativeness = Representativeness_IRS

            else:
                #st.write("Data representativeness assessment is essential to ensure AI reliability. Please perform necessary data representativeness assessment.")
                #response6 = st.selectbox("Do you assess data representativeness?", ('Yes', 'No'), index=None, placeholder="Please select...") 
                fig = fig_individual_score(round(st.session_state.Data_Representativeness*100,2), 'Data Representativeness score is ')
                st.plotly_chart(fig, use_container_width=True) 

    # ---------------------------------------------------------------------------------------------
    if view_button == 'Model Performance Assessment':
        st.subheader("Performance Evaluation")
        
        response7=st.selectbox("Do you evaluate the model performance? ", ('Yes', 'No'), index=None, placeholder="Please select...")
        if response7 == "Yes":
            
            response8 = st.multiselect("Select the performance evaluation metrics ", ('Accuracy Score','AUC','Precision Score'), 
                                       max_selections=3, placeholder="Please select...")
            if ('Accuracy Score' in response8) & ('AUC' not in response8) & ('Precision Score' not in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=True, AUC=False, Precision_Score=False)
            if ('AUC' in response8) & ('Accuracy Score' not in response8) & ('Precision Score' not in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=False, AUC=True, Precision_Score=False)
            if ('Precision Score' in response8) & ('Accuracy Score' not in response8) & ('AUC' not in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=False, AUC=False, Precision_Score=True)
            if ('Accuracy Score' in response8) & ('AUC' in response8) & ('Precision Score' not in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=True, AUC=True, Precision_Score=False)
            if ('Accuracy Score' in response8) & ('Precision Score' in response8) & ('AUC' not in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=True, AUC=False, Precision_Score=True)
            if ('AUC' in response8) & ('Precision Score' in response8) & ('Accuracy Score' not in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=False, AUC=True, Precision_Score=True)
            if ('Accuracy Score' in response8) & ('AUC'  in response8) & ('Precision Score' in response8):
                IRS_in, IRS_off = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test, 
                                                                                      Accuracy_Score=True, AUC=True, Precision_Score=True)
            
            response9=st.selectbox("Do you do some in-sample performance evaluation? ", ('Yes', 'No'), index=None, placeholder="Please select...")
            if response9 == "Yes":
                #st.write(f"The in-sample performance score is {round(100 *IRS_in, 2)}%")

                fig = fig_individual_score(round(IRS_in*100,2), 'In-sample performance score is ')
                st.plotly_chart(fig, use_container_width=True)  
                st.session_state.IRS_In = IRS_in
                
                response10=st.selectbox("Do you do some out-of-sample performance evaluation? ", ('Yes', 'No'), index=None, placeholder="Please select...")
                if response10 == "Yes":
                    fig = fig_individual_score(round(IRS_off*100,2), 'Out of-sample performance score is ')
                    st.plotly_chart(fig, use_container_width=True)   
                    st.session_state.IRS_Off = IRS_off
                    #st.write(f"The out-of-sample performance score is {round(100 *IRS_off, 2)}%")

            else: 
                response10=st.selectbox("Do you do some out-of-sample performance evaluation? ", ('Yes', 'No'), index=None, placeholder="Please select...")
                if response10 == "Yes":
                    #st.write(f"The out-of-sample performance score is {round(100 *IRS_off, 2)}%")
                    fig = fig_individual_score(round(IRS_off*100,2), 'Out of-sample performance score is ')
                    st.plotly_chart(fig, use_container_width=True) 
                    st.session_state.IRS_Off = IRS_off

                else:
                    st.write("Model performance evaluation is essential to ensure AI reliability. Please perform necessary in-sample and out-of-sample performance evaluation.")
                #st.write("**The in sample performance score is {round(100 *IRS_in, 2)}%**")
            
        
        else:
            response8=st.selectbox("Are you using a pre-trained model? ", ('Yes', 'No'), index=None, placeholder="Please select...")
            if response8 == "Yes":
                response9=st.selectbox("Great! Do you benchmark your model? ", ('Yes', 'No'), index=None, placeholder="Please select...")
                if response9 == "Yes":
                    st.write("Awesome!")
                else:
                    st.write("It is important to bechmark your pre-trained model")
            else:
                st.write("YOU HAVE TO EVALUATE THE PERFORMANCE OF YOUR MODEL BEFORE SENDING IT TO VALIDATION TEAM")
            
    # ---------------------------------------------------------------------------------------------
        st.subheader("Robustness checking")
        
        response=st.selectbox("Do you implement a robustness checking?", ('Yes', 'No'), index=None, placeholder="Please select...")
        if response == "Yes":
            if st.session_state.Robutsness_Score == []:
                response=st.selectbox("Does your model employ Neural Networks?", ('Yes', 'No'), index=None, placeholder="Please select...")
                if response == 'Yes':
                    Robutsness_score = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                          st.session_state.x_train, st.session_state.x_test, 
                                                                                          st.session_state.y_train, st.session_state.y_test,
                                                            Accuracy_Score=True, AUC=False, Precision_Score=False, Robustness=True, Keras=True)[2]
                    fig = fig_individual_score(round(Robutsness_score*100,2), 'The robustness checking score is ')
                    st.plotly_chart(fig, use_container_width=True) 
                    st.session_state.Robutsness_Score = Robutsness_score
                    
                elif response == 'No':
                    Robutsness_score = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                          st.session_state.x_train, st.session_state.x_test, 
                                                                                          st.session_state.y_train, st.session_state.y_test,
                                                            Accuracy_Score=True, AUC=False, Precision_Score=False, Robustness=True)[2]
                    fig = fig_individual_score(round(Robutsness_score*100,2), 'The robustness checking score is ')
                    st.plotly_chart(fig, use_container_width=True) 
                    st.session_state.Robutsness_Score = Robutsness_score

            else:
                fig = fig_individual_score(round(st.session_state.Robutsness_Score*100,2), 'The robustness checking score is ')
                st.plotly_chart(fig, use_container_width=True) 

        if response == "No":
            st.write("It is important to implement a robustness checking")
    
    # ---------------------------------------------------------------------------------------------
        st.subheader("Uncertainty quantification")

        response=st.selectbox("Do you quantify uncertainty associated with the model?", ('Yes', 'No'), index=None, placeholder="Please select...")
        if response == "Yes":
            if st.session_state.Uncertainty_Score == []:
                IRS_uncertainty = Classification_Models_V5.UQ_Conformal_Prediction(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.y_train,
                                                                                      st.session_state.x_test, st.session_state.y_test,
                                                                                   class_labels=[0,1], n_classes=2, alpha=0.05)
                #st.write(f"The uncertainty score is {round(IRS_uncertainty, 2)}%")
                fig = fig_individual_score(round(IRS_uncertainty*100,2), 'The uncertainty score is ')
                st.plotly_chart(fig, use_container_width=True) 

                st.session_state.Uncertainty_Score = IRS_uncertainty
            else:
                fig = fig_individual_score(round(st.session_state.Uncertainty_Score*100,2), 'The Uncertainty score is ')
                st.plotly_chart(fig, use_container_width=True)
                                
                                
        if response == "No":
            st.write("It is important to quantify the model uncertainty")
            
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Qualitative Assessment of Reliability':
        st.subheader("Qualitative assessment of reliability")
        
        qualitative_questionnaire()

    # ----------------------------------------------------------------------------------------------------
    if view_button == 'Business Relevance':
        st.subheader("Business Impact Evaluation")
        
        business_impact()
        
# -------------Button-------------------------------------------------------------------------------------------------------
def HML(level):
    if level == "L":
        st.write(f"The business impact is low (L)")
    if level == "M":
        st.write(f"The business impact is Medium (M)")        
    if level == "H":
        st.write(f"The business impact is High (H)")      
    if level == "NA":
        st.write(f"The business impact is Not Applicable (NA)")          
        
def summary_figure(df):
    fig = px.line(df, x='x', y='y', markers=True,range_y=[-1,103]).update_layout(xaxis_title="Dimension", yaxis_title="Reliability Index"
    )

    for i in range(len(df['y'])):
        fig.add_annotation(x=i+0.02, y=df['y'][i]+3,text=str(df['y'][i])+"%",font_size=12, showarrow=False)

    fig.add_hrect(y0=0, y1=50, line_width=4, fillcolor="red", opacity=0.5, annotation_text="Low", annotation_position="left",annotation_font_size=12)
    fig.add_hrect(y0=50, y1=80, line_width=4, fillcolor="orange", opacity=0.5, annotation_text="Medium", annotation_position="left",annotation_font_size=12)
    fig.add_hrect(y0=80, y1=100, line_width=4, fillcolor="green", opacity=0.5, annotation_text="High", annotation_position="left",annotation_font_size=12)
    fig.update_layout(autosize=True,
                      yaxis=dict(showgrid=False,showline=False,showticklabels=False,zeroline=True),
                     paper_bgcolor= "rgba(0, 0, 0, 0)",plot_bgcolor= "rgba(0, 0, 0, 0)")
    return fig

def check_list(testlist, threshold):
    for value in testlist:
        if value < threshold:
            return True
    return False

def Condition_number(Business, Reliability):
    # Create the condition selection function...
    if Reliability=="Low":
        if Business=="NA":
            condition_number="1"
        elif  Business=="L":
            condition_number="2"
        elif Business=="M":
            condition_number="3"
        else:
            condition_number="4"
    elif Reliability=="Medium":
        if Business=="NA":
            condition_number="5"
        elif  Business=="L":
            condition_number="6"
        elif Business=="M":
            condition_number="7"
        else:
            condition_number="8"
    else:
        if Business=="NA":
            condition_number="9"
        elif  Business=="L":
            condition_number="10"
        elif Business=="M":
            condition_number="11"
        else:
            condition_number="12"
    return condition_number

# Visualization in a plan......
image_files = ["Final Output\low-na.jpg", 
           "Final Output\low-low.jpg", 
           "Final Output\low-medium.jpg", 
           "Final Output\low-high.jpg", 
           "Final Output\medium-na.jpg",
           "Final Output\medium-low.jpg", 
           "Final Output\medium-medium.jpg",
           "Final Output\medium-high.jpg",
           "Final Output\high-na.jpg",
           "Final Output\high-low.jpg",
           "Final Output\high-medium.jpg", 
           "Final Output\high-high.jpg"]


image_files_2 = ["Final Output-White\low-na.jpg", 
           "Final Output-White\low-low.jpg", 
           "Final Output-White\low-medium.jpg", 
           "Final Output-White\low-high.jpg", 
           "Final Output-White\medium-na.jpg",
           "Final Output-White\medium-low.jpg", 
           "Final Output-White\medium-medium.jpg",
           "Final Output-White\medium-high.jpg",
           "Final Output-White\high-na.jpg",
           "Final Output-White\high-low.jpg",
           "Final Output-White\high-medium.jpg", 
           "Final Output-White\high-high.jpg"]


conditions = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

st.sidebar.markdown("***")
if st.sidebar.button('Show summary', type='secondary', use_container_width=True):

    df = pd.DataFrame(dict(
        x = ['Data<br>Avalability','Data<br>Quality', 'Data<br>Rrepresentativeness',
             'In-sample<br>Performance','Out-of-sample<br>Performance','Robustness<br>Score','Uncertainty<br>Score',
             'Qualitative<br>Assessment'],
        y = [round(100*st.session_state.Data_Availability, 2), 
             round(100*st.session_state.Data_Quality,2),
             round(100*st.session_state.Data_Representativeness,2),
             round(100*st.session_state.IRS_In, 2),
             round(100*st.session_state.IRS_Off, 2),
             round(100*st.session_state.Robutsness_Score, 2),
             round(100*st.session_state.Uncertainty_Score, 2),
             round(100*st.session_state.QS, 2)]
        ))
    fig = summary_figure(df)
    st.plotly_chart(fig, use_container_width=True)  
 

if st.sidebar.button('Calculate score', type='secondary', use_container_width=True):
    st.subheader("Final aggregation")
    # # Import the user-defined AHP weight matrix
    # Weight_AHP = pd.read_excel('Weight_AHP.xlsx',index_col=0)
    # weight_AHP = Weight_AHP.to_numpy()
    # # convert it into weight vector
    # st.session_state.weight_vector = Final_Aggregation.Agg_weight(weight_AHP)
    
    # consolidate score_vector
    score_vector = []
    dimension_score = [st.session_state.Data_Availability, st.session_state.Data_Quality, st.session_state.Data_Representativeness,   
                    st.session_state.IRS_In, st.session_state.IRS_Off, st.session_state.Robutsness_Score, st.session_state.Uncertainty_Score]
    for dim in dimension_score:
        score_vector.append(dim)

    Reliability = []
    Reliability_Index=Final_Aggregation.Reliability_Index(st.session_state.weight_vector, score_vector, st.session_state.QS)
    st.write(f"The Model Reliability Index is {round(Reliability_Index[0], 2)}%")
    st.write(f"The Model Reliability Index level is {Reliability_Index[1]}")
    Reliability.append(Reliability_Index[1])
    
    
    if Reliability!=[]:
        selected_condition = Condition_number(st.session_state.Business, Reliability[0])
        # Find the index of the selected condition
        condition_index = conditions.index(selected_condition)
        # Load and display the corresponding image
        image = Image.open(f"{image_files[condition_index]}") # Adjust the file extension if necessary
        # st.image(image, caption=image_files[condition_index], use_column_width=True) 
        st.image(image, use_column_width=True) 
        

    # show weakness message
    if check_list(dimension_score, 0.8):
    
        col_weak, _ = st.columns([3,1])
        with col_weak:
            with st.container(border=True):
                st.markdown(
                    """
                    <style>
                    .weak-style {
                        background-color: rgb(241, 92, 92);
                        padding: 10px;
                        margin-bottom: 10px;
                        width: 90%;
                        color: black;
                        font-bold: bold;
                    }
                    .red-subheader {
                        color: rgb(238, 90,90);
                        font-weight: bold;
                        font-size: 22px;
                        margin-top:0px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<span class='red-subheader'>Weaknesses:</span>", unsafe_allow_html=True)
                
                if (st.session_state.Data_Availability<0.8)|(st.session_state.Data_Quality<0.8)|(st.session_state.Data_Representativeness<0.8):
                    st.markdown(
                        """
                        <div class="weak-style">
                        ❌ <strong>Ineffective data governance:</strong> Requires a mitigation plan to improve data availability, data quality, and data representativeness.
                        </div>
                        """, unsafe_allow_html=True)
                if (st.session_state.IRS_In<0.8)|(st.session_state.IRS_Off<0.8)|(st.session_state.Robutsness_Score<0.8)|(st.session_state.Uncertainty_Score<0.8):
                    st.markdown(
                        """
                        <div class="weak-style">
                        ❌ <strong>Poor model performance:</strong> Requires model recalibration or retraining to improve model effectiveness.
                        </div>
                        """, unsafe_allow_html=True)
                if abs(st.session_state.IRS_In - st.session_state.IRS_Off)>0.1:
                    st.markdown(
                        """
                        <div class="weak-style">
                        ❌ <strong>Over/under fitting:</strong> Requires a remediation strategy and careful tuning of model complexity and hyperparameters to build robust models.
                        </div>
                        """, unsafe_allow_html=True)           
                        
                    

            
def Questionnaire_calculation(Ask, counts_applicable, counts_yes):
    if Ask=="Yes":
        counts_yes = counts_yes+1
        counts_applicable = counts_applicable
    elif Ask=="No":
        counts_yes = counts_yes
        counts_applicable = counts_applicable
    elif Ask=="NA":
        counts_yes = counts_yes
        counts_applicable = counts_applicable - 1
    return counts_yes, counts_applicable

    

# Define the reassessment function to toggle the reassessment state
def reassessment_summary():
    # # st.session_state.reassessment_active = False
    # st.session_state.reassessment_active = not st.session_state.reassessment_active
    st.subheader("Reassessment")
    # st.markdown('Please navigate to **Upload File** page to submit new files for reassessment.')
    # Availability_IRS_2 = Data_Availability.Availability_IRS(num_var_needed=st.session_state.num_var, df = st.session_state.training_df, prct=0.4)
    # Quality_IRS_2 = Data_Quality_V2.Data_Quality_Score(st.session_state.training_df)
    # Representativeness_IRS_2=Data_Representativeness_V2.Data_representativeness_score(st.session_state.training_df, 
    #                                                                                 st.session_state.ref_df, 
    #                                                                                 significant=0.05)
    # IRS_in_2, IRS_off_2 = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
    #                                                                       st.session_state.x_train, st.session_state.x_test, 
    #                                                                       st.session_state.y_train, st.session_state.y_test, 
    #                                                                       Accuracy_Score=True, AUC=True, Precision_Score=True)
    ### need to find a way to specify whether neural network model or not
    # Robutsness_score_2 = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
    #                                                                       st.session_state.x_train, st.session_state.x_test, 
    #                                                                       st.session_state.y_train, st.session_state.y_test,
    #                                         Accuracy_Score=True, AUC=False, Precision_Score=False, Robustness=True, Keras=True)[2]
    # Uncertainty_2 = Classification_Models_V5.UQ_Conformal_Prediction(st.session_state.pretrained_model, 
    #                                                                       st.session_state.x_train, st.session_state.y_train,
    #                                                                       st.session_state.x_test, st.session_state.y_test,
    #                                                                    class_labels=[0,1], n_classes=2, alpha=0.05)

    
# ----------------------------------------------------------------------------------
    st.subheader("Data availability assessment")        
    response1 = st.selectbox("Do you ensure the data availability? ", ('Yes', 'No'))      
    if response1 == 'Yes':
        Availability_IRS_2 = Data_Availability.Availability_IRS(num_var_needed=st.session_state.num_var, df = st.session_state.training_df, prct=0.4)
        fig = fig_individual_score(round(Availability_IRS_2*100,2), 'Data Avalability score is ')
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Data quality assessment")
    response2 = st.selectbox("Do you assess data quality?", ('Yes', 'No'))
    if response2 == 'Yes':
        Data_Quality_temp1= Data_Quality_V2.Coherence_IRS(st.session_state.training_df)
        fig = fig_individual_score(round(Data_Quality_temp1*100,2), 'Data coherence score is ')
        st.plotly_chart(fig, use_container_width=True)
        
        Data_Quality_temp2 = Data_Quality_V2.Completeness_IRS(st.session_state.training_df)
        fig = fig_individual_score(round(Data_Quality_temp2*100,2), 'Data completeness score is ')
        st.plotly_chart(fig, use_container_width=True)
        
        Data_Quality_temp3 = Data_Quality_V2.Uniqueness_IRS(st.session_state.training_df)
        fig = fig_individual_score(round(Data_Quality_temp3*100,2), 'Data uniqueness score is ')
        st.plotly_chart(fig, use_container_width=True)
        
        Quality_IRS_2 = min(Data_Quality_temp1,Data_Quality_temp2,Data_Quality_temp3)
        fig = fig_individual_score(round(Quality_IRS_2*100,2), 'Data quality score is ')
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Data representativeness assessment") 
    response6 = st.selectbox("Do you assess data representativeness?", ('Yes', 'No'))     
    if response6 == 'Yes':
        Representativeness_IRS_2=Data_Representativeness_V2.Data_representativeness_score(st.session_state.training_df, 
                                                                                st.session_state.ref_df, 
                                                                                significant=0.05)
        fig = fig_individual_score(round(Representativeness_IRS_2*100,2), 'Data representativeness score is ')
        st.plotly_chart(fig, use_container_width=True)     

    st.subheader("Performance Evaluation")
    response7=st.selectbox("Do you evaluate the model performance? ", ('Yes', 'No'))
    if response7 == "Yes":    
        IRS_in_2, IRS_off_2 = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                              st.session_state.x_train, st.session_state.x_test, 
                                                                              st.session_state.y_train, st.session_state.y_test, 
                                                                              Accuracy_Score=True, AUC=True, Precision_Score=True)
        Robutsness_score_2 = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                      st.session_state.y_train, st.session_state.y_test,
                                        Accuracy_Score=True, AUC=False, Precision_Score=False, Robustness=True, Keras=True)[2]
        response9=st.selectbox("Do you do some in-sample performance evaluation? ", ('Yes', 'No'))
        if response9 == "Yes":
            fig = fig_individual_score(round(IRS_in_2*100,2), 'In-sample performance score is ')
            st.plotly_chart(fig, use_container_width=True)  
            
        response10=st.selectbox("Do you do some out-of-sample performance evaluation? ", ('Yes', 'No'))
        if response10 == "Yes":
            fig = fig_individual_score(round(IRS_off_2*100,2), 'Out of-sample performance score is ')
            st.plotly_chart(fig, use_container_width=True)   
            
    response=st.selectbox("Do you implement a robustness checking?", ('Yes', 'No'))
    if response == "Yes":
        fig = fig_individual_score(round(Robutsness_score_2*100,2), 'The robustness checking score is ')
        st.plotly_chart(fig, use_container_width=True)         

    response=st.selectbox("Do you quantify uncertainty associated with the model?", ('Yes', 'No'))
    if response == "Yes":
        Uncertainty_2 = Classification_Models_V5.UQ_Conformal_Prediction(st.session_state.pretrained_model, 
                                                                      st.session_state.x_train, st.session_state.y_train,
                                                                      st.session_state.x_test, st.session_state.y_test,
                                                                    class_labels=[0,1], n_classes=2, alpha=0.05)
        fig = fig_individual_score(round(Uncertainty_2*100,2), 'The uncertainty score is ')
        st.plotly_chart(fig, use_container_width=True) 
    
    st.subheader("Qualitative assessment of reliability")
    response=st.selectbox("Is your model intended for prediction/classification purpose? ", ["Yes", "No"])
    if response=="Yes":
        counts_applicable = 20
        counts_yes = 0
        Ask = st.selectbox("Do you ensure data availability?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you test the dataset for representativeness? ", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you check the data quality?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you apply a data quality issue mitigation strategy?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Aptness of model choice with business use and target data?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you test your model for over/under-fitting identification and remediation?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you perform an in-sample performance evaluation?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you perform an out-of-sample performance evaluation?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Robustness checking?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Benchmarking of your model?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Sensitivity analysis?", ["No","Yes", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you perform uncertainty quantification for your model?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you regularly check the production data quality?", ["NA","Yes", "No"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you regularly test the production data for data drift issues?", ["NA","Yes", "No"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you regularly monitor the model performance drift?", ["NA","Yes", "No"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you regularly monitor the model output stability?", ["NA","Yes", "No"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you regularly update your model benchmarking?", ["NA","Yes", "No"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Do you regularly collect users’ feedback & suggestions for future improvement of your model?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Is an action plan defined for the model retrain/recalibration/review?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)
        Ask = st.selectbox("Is there a fall-back mechanism in case the AI system is not performing as expected during production?", ["Yes", "No", "NA"])
        counts_yes, counts_applicable = Questionnaire_calculation(Ask, counts_applicable, counts_yes)        

        if counts_applicable!=0:
            qs = counts_yes/counts_applicable
            st.write(f" The qualitative score is {round(100 * qs, 2)}%.")    
    
    st.subheader("Summary of new reliability score for each dimension") 
    df_2 = pd.DataFrame(dict(
        x = ['Data<br>Avalability','Data<br>Quality', 'Data<br>Rrepresentativeness',
              'In-sample<br>Performance','Out-of-sample<br>Performance','Robustness<br>Score','Uncertainty<br>Score',
              'Qualitative<br>Assessment'],
        y = [round(100*Availability_IRS_2, 2), 
              round(100*Quality_IRS_2,2),
              round(100*Representativeness_IRS_2,2),
              round(100*IRS_in_2, 2),
              round(100*IRS_off_2, 2),
              round(100*Robutsness_score_2, 2),
              round(100*Uncertainty_2, 2),
              round(100*qs, 2)]
        ))
    fig = summary_figure(df_2)
    st.plotly_chart(fig, use_container_width=True)  
    
    st.subheader("Final reliability index") 
    score_vector_2 = [Availability_IRS_2, Quality_IRS_2, Representativeness_IRS_2, IRS_in_2, IRS_off_2, Robutsness_score_2, Uncertainty_2]
    Reliability_Index_2=Final_Aggregation.Reliability_Index(st.session_state.weight_vector, score_vector_2, qs)
    st.write(f"The new model reliability index is {round(Reliability_Index_2[0], 2)}%")
    st.write(f"The new model reliability index level is {Reliability_Index_2[1]}")
    
    selected_condition_2 = Condition_number(st.session_state.Business, Reliability_Index_2[1])
    # Find the index of the selected condition
    condition_index = conditions.index(selected_condition_2)
    # Load and display the corresponding image
    image = Image.open(f"{image_files[condition_index]}") # Adjust the file extension if necessary
    # st.image(image, caption=image_files[condition_index], use_column_width=True) 
    st.image(image, use_column_width=True) 

if "reassessment_active" not in st.session_state:
    st.session_state.reassessment_active = False
        
if st.sidebar.button('Reassessment', type='secondary', use_container_width=True):
    st.session_state.reassessment_active = not st.session_state.reassessment_active

    reassessment_summary()















