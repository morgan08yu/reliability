# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:56:12 2024

@author: YR998EN
"""


import pickle
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import plotly.graph_objects as go
import plotly.express as px

# other useful package used by python.
import tempfile
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


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
#from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

#from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image
#from reportlab.lib import colors
#from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import PageBreak


# import plotly.io as pio
import io
from io import BytesIO
import os
# python functions used....


# import functions pre-defined in other ipynb files 
# import import_ipynb
# make sure this file and those fuction ipynb files are in the same folder, otherwise specifies the path of function files
import Data_Quality_V2
import Data_Availability
import Data_Representativeness_V2
import Classification_Models_V5
# import Conformal_prediction_UQ_multiclass_classification
import Final_Aggregation

#Regression
import Performance_Rob_UQ_regression_v4

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
if "score_vector" not in st.session_state:
    st.session_state.score_vector = []
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
if "Ongoing_Monitoring" not in st.session_state:
    st.session_state.Ongoing_Monitoring = []

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
    
    

if "user" not in st.session_state:
    st.session_state.user=[]
if "user1" not in st.session_state:
    st.session_state.user1=[]
if "critical" not in st.session_state:
    st.session_state.critical=[]
    
if "Indexs" not in st.session_state:
    st.session_state.Indexs=[]
    


if "figsummary" not in st.session_state:  
    st.session_state.figsummary = []
if "reliability_index_image" not in st.session_state:  
    st.session_state.reliability_index_image = []
if "availability_fig" not in st.session_state: 
    st.session_state.availability_fig = []
if "data_quality_fig" not in st.session_state: 
    st.session_state.data_quality_fig = []
if "data_rep_fig" not in st.session_state: 
    st.session_state.data_rep_fig = []
if "insample_fig" not in st.session_state: 
    st.session_state.insample_fig = []
if "offsample_fig" not in st.session_state: 
    st.session_state.offsample_fig = []
if "robustness_fig" not in st.session_state: 
    st.session_state.robustness_fig = []
if "uncertainty_fig" not in st.session_state: 
    st.session_state.uncertainty_fig = []
# ------------------------------------------------------------------------------
def fig_individual_score1(score,dimension):
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

def fig_individual_score2(score, dimension):
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    
    ax.barh('Reliability Index', 50, color='#c91111cc', label='Low', height=0.087)
    ax.barh('Reliability Index', 30, color='orange', label='Medium', left=50, height=0.087)
    ax.barh('Reliability Index', 20, color='#11c954cc', label='High', left=80, height=0.087)
    
    ax.axvline(x=score, color='#7d7d7d', linestyle='--', linewidth=1.5, ymin=0.3, ymax=0.7)
    ax.text(score-25, 0.26, f"{dimension}{score}%.", verticalalignment='top', fontsize=8)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ylabel = ax.set_ylabel(r'Reliability' + '\n' + r'Index', fontsize=8,rotation=0,labelpad=20)
    ylabel.set_y(0.43)
    ax.legend(loc='lower center', fontsize=8, ncol=3, bbox_to_anchor=(0.5, 0.1))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    
    fig.patch.set_facecolor('none')
    # fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('none')
    
    plt.tight_layout() 
    
    return fig

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
  
@st.cache_resource
def upload_file_section():
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
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
        with open(tmp_file.name, 'rb') as file:
            pretrained_model = pickle.load(file)
        # pretrained_model = pickle.load(open(uploaded_model.name, 'rb'))
        st.session_state.pretrained_model = pretrained_model
    
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
            st.success('A reference dataset has been created by sampling the training and testing data to reflect the patterns of complete dataset. This helps simulate the real-world scenarios that the model is expected to encounter.')
            # st.markdown("""<style>
            #             .message-style {background-color: rgb(204,255,179);
            #                             padding: 10px;
            #                             margin-bottom: 10px;
            #                             width: 90%;
            #                             color: black;
            #                             font-bold: bold;}
            #             </style>
            #             """,unsafe_allow_html=True)
            # st.markdown("""<div class="message-style">
            #     A reference dataset has been created by sampling the training and testing data to reflect the patterns of complete dataset. This helps simulate the real-world scenarios that the model is expected to encounter.
            #     </div>
            #     """, unsafe_allow_html=True)
    
    if (uploaded_model is not None) & (uploaded_data1 is not None) & (uploaded_data2 is not None)  :
        response = st.selectbox("Are there any additional transformations required for your data before fitting it into the model?", ('Yes','No'), index=None, 
                                help="For example, converting 2D data to 3D to better fit the model's requirements.",
                                placeholder='Please select...') 
        if response =='Yes':
            trans_data = st.file_uploader(label=":page_facing_up: Upload the transformed data", type=["pkl"],
                                          help="This PKL file should include the following elements in this order: (1) transformed training data for explanatory variables, (2) transformed testing data for explanatory variables, (3) transformed training data for target variables, and (4) transformed testing data for target variables.")
            if trans_data is not None:
                data = []
                with open(trans_data.name, 'rb') as f:
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
                
@st.cache_resource             
def data_availability_section():
    st.subheader("Data availability assessment")        
    response1 = st.selectbox("Do you ensure the data availability? ", ('Yes', 'No'), index=None, placeholder="Please select...",
                             help='Data availability calculates the percentage of available variables within the set of variables needed to train the model. When the missing values rate of a variable is above an acceptable threshold, it will be considered as unavailable.') 
    if response1 == 'No':
        st.write("Data availability assessment is essential to ensure AI reliability. Please perform necessary data availability assessment.")      
    if response1 == 'Yes':
        num_var = st.number_input("How many variables will you need for the analysis and model training, including both explanatory (input) and target (output) variables?", 
                                 min_value=1, value=None, 
                                 help='Explanatory variables are factors that might influence or predict your outcomes. Target variables are the outcomes you want to predict.',
                                 placeholder="Type a number...")
        if num_var is not None:
            Availability_IRS = Data_Availability.Availability_IRS(num_var_needed=num_var, df = st.session_state.training_df, prct=0.4)
            #st.write(f"The Data Availability Index is {round(100*Availability_IRS, 2)}%")
            fig = fig_individual_score1(round(Availability_IRS*100,2), 'Data Avalability score is ')
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.Data_Availability = Availability_IRS
            st.session_state.availability_fig = fig

@st.cache_resource
def data_quality_section():
    Data_Quality_options2 = ['1', '2','3']
    Data_Quality_options3 = ['Data Coherence', 'Data Completeness', 'Data Uniqueness']
    
    st.subheader("Data quality assessment")
    response2 = st.selectbox("Do you assess data quality?", ('Yes', 'No'), index=None, placeholder="Please select...",
                             help='Data Quality assesses the fitness of the data for use. It is evaluated based on various sub-dimensions such as coherence, completeness, and uniqueness. The final data quality score is derived by aggregating these individual scores.')
    if response2 == 'No':
        st.write("Data quality assessment is essential to ensure AI reliability. Please perform necessary data quality assessment.") 
    if response2 == 'Yes':
        response3 = st.selectbox("How many dimensions have you considered in data quality assessment?", Data_Quality_options2, 
                                 index=None, placeholder="Please select...")
        if response3 is not None:
            response3 = int(response3)
            if response3 == 1:
                response4 = st.selectbox("Select the dimension that have been considered in data quality assessment", Data_Quality_options3, 
                                         index=None, placeholder="Please select...",
                                         help='(1) Data coherence checks for anomalies or outliers. (2) Data completeness assesses sufficiency of information. (3) Data uniqueness checks for duplicates. ')
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
                    fig = fig_individual_score1(round(Data_Quality_temp*100,2), response4+' score is ')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.Data_Quality = Data_Quality_temp
                    st.session_state.data_quality_fig = fig
                    
            if response3 > 1:
                response5 = st.multiselect("Select the dimension that have been considered in data quality assessment", Data_Quality_options3, 
                                           max_selections=response3, placeholder="Please select...",
                                           help='(1) Data coherence checks for anomalies or outliers. (2) Data completeness assesses sufficiency of information. (3) Data uniqueness checks for duplicates. ')       
                Data_Quality2 = []
                if 'Data Coherence' in response5:
                    Data_Quality_Coherence = Data_Quality_V2.Coherence_IRS(st.session_state.training_df)
                    #st.write(f"The Data Coherence score is {round(100*Data_Quality_Coherence, 2)}%")
                    Data_Quality2.append(Data_Quality_Coherence)
                    fig = fig_individual_score1(round(Data_Quality_Coherence*100,2), 'Data Coherence score is ')
                    st.plotly_chart(fig, use_container_width=True)  
                if 'Data Completeness' in response5:
                    Data_Quality_Completeness = Data_Quality_V2.Completeness_IRS(st.session_state.training_df)
                    #st.write(f"The Data Completeness score is {round(100*Data_Quality_Completeness, 2)}%")
                    Data_Quality2.append(Data_Quality_Completeness)
                    fig = fig_individual_score1(round(Data_Quality_Completeness*100,2), 'Data Completeness score is ')
                    st.plotly_chart(fig, use_container_width=True)   
                if 'Data Uniqueness' in response5:
                    Data_Quality_Uniqueness = Data_Quality_V2.Uniqueness_IRS(st.session_state.training_df)
                    #st.write(f"The Data Uniqueness score is {round(100*Data_Quality_Uniqueness, 2)}%")
                    Data_Quality2.append(Data_Quality_Uniqueness)
                    fig = fig_individual_score1(round(Data_Quality_Uniqueness*100,2), 'Data Uniqueness score is ')
                    st.plotly_chart(fig, use_container_width=True)
                if Data_Quality2 != []:
                    Data_Quality_temp = min(Data_Quality2)
                    st.write("Below is the aggregated Data Quality score by taking the minimum of individual scores above.") 
                    fig = fig_individual_score1(round(Data_Quality_temp*100,2), 'Data Quality score is ')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.Data_Quality = Data_Quality_temp
                    st.session_state.data_quality_fig = fig
                    
@st.cache_resource
def data_representativeness_section():
    st.subheader("Data representativeness assessment") 
    response6 = st.selectbox("Do you assess data representativeness?", ('Yes', 'No'), index=None, placeholder="Please select...",
                             help='Data Representativeness assesses if the training data sample is representative of the full dataset for both numerical and categorical variables.') 
    if response6 == 'No':
        st.write("Data representativeness assessment is essential to ensure AI reliability. Please perform necessary data representativeness assessment.")      
    if response6 == 'Yes':
        if st.session_state.Data_Representativeness == []:
            Representativeness_IRS=Data_Representativeness_V2.Data_representativeness_score(st.session_state.training_df, 
                                                                                            st.session_state.ref_df, 
                                                                                            significant=0.05)
            #st.write(f"The Data Representativeness Index is {round(100*Representativeness_IRS, 2)}%")
            fig = fig_individual_score1(round(Representativeness_IRS*100,2), 'Data Representativeness score is ')
            st.plotly_chart(fig, use_container_width=True)    
            st.session_state.Data_Representativeness = Representativeness_IRS
            st.session_state.data_rep_fig = fig
        else:
            fig = fig_individual_score1(round(st.session_state.Data_Representativeness*100,2), 'Data Representativeness score is ')
            st.plotly_chart(fig, use_container_width=True) 

@st.cache_resource    
def performance_evaluation_section(classification=False,regression=False, clustering=False,generativeAI=False):
    st.subheader("Performance Evaluation")
    response7=st.selectbox("Do you evaluate the model performance? ", ('Yes', 'No'), index=None, placeholder="Please select...")
    if response7 == "Yes":
        if classification:
            response8 = st.multiselect("Select the performance evaluation metrics ", ('Accuracy Score','AUC','Precision Score'), 
                                   max_selections=3, placeholder="Please select...",
                                   help='When multiple metrics are selected, the final performance score is derived by aggregating these metrics. ')
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
        if regression:
            response8 = st.multiselect("Select the performance evaluation metrics ", ('Relative Absolute Error','Coefficient of determination','Mean Absolute Error', 'Normalized RMSE'),
                                       max_selections=4, placeholder="Please select...",
                                       help='When multiple metrics are selected, the final performance score is derived by aggregating these metrics. ')


            if ('Relative Absolute Error' in response8) & ('MAPE' not in response8) & ('Mean Absolute Error' not in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=False, NRMSE=False, Robustness=False, R_2=False, Keras=False, num_folds=3)

            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' not in response8) & ('Mean Absolute Error' not in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=False, NRMSE=False, Robustness=False, R_2=True, Keras=False, num_folds=3)

            if ('Mean Absolute Error' in response8) & ('Relative Absolute Error' not in response8) & ('MAPE' not in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=True, NRMSE=False, Robustness=False, R_2=False, Keras=False, num_folds=3)

            if ('Normalized RMSE' in response8) & ('Relative Absolute Error' not in response8) & ('Mean Absolute Error' not in response8) & ('MAPE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=False, NRMSE=True, Robustness=False, R_2=False, Keras=False, num_folds=3)

            #2 by 2
            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' not in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=False, NRMSE=False, Robustness=False, R_2=True, Keras=False, num_folds=3)

            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' not in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=True, NRMSE=False, Robustness=False, R_2=True, Keras=False, num_folds=3)

            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' not in response8) & ('Mean Absolute Error' not in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=False, NRMSE=True, Robustness=False, R_2=True, Keras=False, num_folds=3)

            if ('Coefficient of determination' not in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=True, NRMSE=False, Robustness=False, R_2=False, Keras=False, num_folds=3)

            if ('Coefficient of determination' not in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' not in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=False, NRMSE=True, Robustness=False, R_2=False, Keras=False, num_folds=3)

            if ('Coefficient of determination' not in response8) & ('Relative Absolute Error' not in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=True, NRMSE=True, Robustness=False, R_2=False, Keras=False, num_folds=3)


            # 3 by 3
            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' not in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=True, NRMSE=False, Robustness=False, R_2=True, Keras=False, num_folds=3)

            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' not in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=False, NRMSE=True, Robustness=False, R_2=True, Keras=False, num_folds=3)

            if ('Coefficient of determination' not in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=True, NRMSE=True, Robustness=False, R_2=False, Keras=False, num_folds=3)

            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' not in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=False, MAE=True, NRMSE=True, Robustness=False, R_2=True, Keras=False, num_folds=3)

            #4 selected
            if ('Coefficient of determination' in response8) & ('Relative Absolute Error' in response8) & ('Mean Absolute Error' in response8) & ('Normalized RMSE' in response8):
                IRS_in, IRS_off= Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=True, NRMSE=True, Robustness=False, R_2=True, Keras=False, num_folds=3)
        
        response9=st.selectbox("Do you do some in-sample performance evaluation? ", ('Yes', 'No'), index=None, placeholder="Please select...",
                               help='The in-sample performance evaluation assesses how well the model fits the data it was trained on. ')
        if response9 == "Yes":
            #st.write(f"The in-sample performance score is {round(100 *IRS_in, 2)}%")
            fig = fig_individual_score1(round(IRS_in*100,2), 'In-sample performance score is ')
            st.plotly_chart(fig, use_container_width=True)  
            st.session_state.IRS_In = IRS_in
            st.session_state.insample_fig = fig
            
            response10=st.selectbox("Do you do some out-of-sample performance evaluation? ", ('Yes', 'No'), index=None, placeholder="Please select...",
                                    help='Out-of-sample performance evaluation tests the model on unseen data, providing insight into its generalizability. ')
            if response10 == "Yes":
                fig = fig_individual_score1(round(IRS_off*100,2), 'Out of-sample performance score is ')
                st.plotly_chart(fig, use_container_width=True)   
                st.session_state.IRS_Off = IRS_off
                st.session_state.offsample_fig = fig
                #st.write(f"The out-of-sample performance score is {round(100 *IRS_off, 2)}%")
        else: 
            response10=st.selectbox("Do you do some out-of-sample performance evaluation? ", ('Yes', 'No'), index=None, placeholder="Please select...",
                                    help='Out-of-sample performance evaluation tests the model on unseen data, providing insight into its generalizability. ')
            if response10 == "Yes":
                #st.write(f"The out-of-sample performance score is {round(100 *IRS_off, 2)}%")
                fig = fig_individual_score1(round(IRS_off*100,2), 'Out of-sample performance score is ')
                st.plotly_chart(fig, use_container_width=True) 
                st.session_state.IRS_Off = IRS_off
                st.session_state.offsample_fig = fig
            else:
                st.write("Model performance evaluation is essential to ensure AI reliability. Please perform necessary in-sample and out-of-sample performance evaluation.")
    else:
        response8=st.selectbox("Are you using a pre-trained model? ", ('Yes', 'No'), index=None, placeholder="Please select...")
        if response8 == "Yes":
            response9=st.selectbox("Great! Do you benchmark your model? ", ('Yes', 'No'), index=None, placeholder="Please select...")
            if response9 == "Yes":
                st.write("Awesome!")
            else:
                st.write("It is important to bechmark your pre-trained model")
        elif response8 == "No":
            st.write("You need to evaluate the performance of your model before sending it to validation team.")

@st.cache_resource
def robustness_section(classification=False,regression=False, clustering=False,generativeAI=False):
    st.subheader("Robustness checking")
    response_R=st.selectbox("Do you implement a robustness checking?", ('Yes', 'No'), index=None, placeholder="Please select...", 
                            help='The Robustness checking tests the resilience of models against adversarial attacks. It involves simulating attacks on the model to identify vulnerabilities and weaknesses.')
    if response_R == "Yes":
        if st.session_state.Robutsness_Score == []:
            response=st.selectbox("Does your model employ Neural Networks?", ('Yes', 'No'), index=None, placeholder="Please select...")
            if response == 'Yes':
                if classification:
                    Robutsness_score = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test,
                                                        Accuracy_Score=True, AUC=False, Precision_Score=False, Robustness=True, Keras=True)[2]
                if regression:
                    Robutsness_score = Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=True, NRMSE=True, Robustness=True, R_2=False, Keras=True, num_folds=3)[2]                    
                
                fig = fig_individual_score1(round(Robutsness_score*100,2), 'The robustness checking score is ')
                st.plotly_chart(fig, use_container_width=True) 
                st.session_state.Robutsness_Score = Robutsness_score
                st.session_state.robustness_fig = fig
            elif response == 'No':
                if classification:
                    Robutsness_score = Classification_Models_V5.Performance_Classification(st.session_state.pretrained_model, 
                                                                                      st.session_state.x_train, st.session_state.x_test, 
                                                                                      st.session_state.y_train, st.session_state.y_test,
                                                        Accuracy_Score=True, AUC=False, Precision_Score=False, Robustness=True)[2]
                if regression:
                    Robutsness_score = Performance_Rob_UQ_regression_v4.Performance_Regression(st.session_state.pretrained_model, st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, MAPE=False, RAE=True, MAE=True, NRMSE=True, Robustness=True, R_2=False, Keras=True, num_folds=3)[2]                    
                    
                fig = fig_individual_score1(round(Robutsness_score*100,2), 'The robustness checking score is ')
                st.plotly_chart(fig, use_container_width=True) 
                st.session_state.Robutsness_Score = Robutsness_score
                st.session_state.robustness_fig = fig
        else:
            fig = fig_individual_score1(round(st.session_state.Robutsness_Score*100,2), 'The robustness checking score is ')
            st.plotly_chart(fig, use_container_width=True) 
    if response_R == "No":
        st.write("It is important to implement a robustness checking")

@st.cache_resource
def uncertainty_section(classification=False,regression=False, clustering=False,generativeAI=False):
    st.subheader("Uncertainty quantification")
    response=st.selectbox("Do you quantify uncertainty associated with the model?", ('Yes', 'No'), index=None, placeholder="Please select...", 
                          help='Uncertainty refers to the lack of confidence in the output of a model.')
    if response == "Yes":
        if st.session_state.Uncertainty_Score == []:
            if classification:
                IRS_uncertainty = Classification_Models_V5.UQ_Conformal_Prediction(st.session_state.pretrained_model, 
                                                                                  st.session_state.x_train, st.session_state.y_train,
                                                                                  st.session_state.x_test, st.session_state.y_test,
                                                                               class_labels=[0,1], n_classes=2, alpha=0.05)
            if regression:
                IRS_uncertainty = Performance_Rob_UQ_regression_v4.reg_GPR_ECE(st.session_state.x_train[0:2000,1,:], st.session_state.x_test[:,1,:], 
                                                                               st.session_state.y_train[0:2000], st.session_state.y_test,
                                                     K=100)
            #st.write(f"The uncertainty score is {round(IRS_uncertainty, 2)}%")
            fig = fig_individual_score1(round(IRS_uncertainty*100,2), 'The uncertainty score is ')
            st.plotly_chart(fig, use_container_width=True) 
            st.session_state.Uncertainty_Score = IRS_uncertainty
            st.session_state.uncertainty_fig = fig
        else:
            fig = fig_individual_score1(round(st.session_state.Uncertainty_Score*100,2), 'The Uncertainty score is ')
            st.plotly_chart(fig, use_container_width=True)
                            
    if response == "No":
        st.write("It is important to quantify the model uncertainty")

@st.cache_resource
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
        st.session_state.user=user_input
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

@st.cache_resource
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
    st.session_state.user1=user_input
    count_h = sum(1 for value in user_input.values() if value == "H")
    count_m = sum(1 for value in user_input.values() if value == "M")
    count_l = sum(1 for value in user_input.values() if value == "L")
    
    response=st.selectbox("Is the AI use cases involved in life-critical or safety-critical operations?", ["Yes", "No"], index=None, placeholder="Please select...")
    
    st.session_state.critical=response 
    
    if response=="Yes" or count_h !=0:
        st.write("The Business Impact is High (H)")
        st.session_state.Business="H"
    elif count_m !=0:
        st.write("The Business Impact is Medium (M)")
        st.session_state.Business="M"
    elif count_l !=0:
        st.write("The Business Impact is Low (L)")
        st.session_state.Business="L"
    else:
        st.write("The Business Impact is Not Applicable (NA)")
        st.session_state.Business="NA"


# --------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
if model_selector == 'Classification':
    
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Upload File':
        upload_file_section()
                
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Data Assessment':

        data_availability_section()
        
        data_quality_section() 
         
        data_representativeness_section()

    # ---------------------------------------------------------------------------------------------
    if view_button == 'Model Performance Assessment':
        
        performance_evaluation_section(classification=True,regression=False, clustering=False,generativeAI=False)

        robustness_section(classification=True,regression=False, clustering=False,generativeAI=False)
        
        uncertainty_section(classification=True,regression=False, clustering=False,generativeAI=False)

    # ---------------------------------------------------------------------------------------------
    if view_button == 'Ongoing Monitoring':    
        st.header("_Coming soon..._")    
        
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Qualitative Assessment of Reliability':
        
        st.subheader("Qualitative assessment of reliability")
        qualitative_questionnaire()

    # ----------------------------------------------------------------------------------------------------
    if view_button == 'Business Relevance':
        
        st.subheader("Business Impact Evaluation")
        business_impact()

# ---------------------------------------------------------------------------------------------

if model_selector == 'Regression':
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Upload File':
        
        upload_file_section()
                
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Data Assessment':
        
        data_availability_section()
         
        data_quality_section() 
                           
        data_representativeness_section()

    # ---------------------------------------------------------------------------------------------
    if view_button == 'Model Performance Assessment':
        
        performance_evaluation_section(classification=False,regression=True, clustering=False,generativeAI=False)

        robustness_section(classification=False,regression=True, clustering=False,generativeAI=False)
    
        uncertainty_section(classification=False,regression=True, clustering=False,generativeAI=False)

    # ---------------------------------------------------------------------------------------------
    if view_button == 'Ongoing Monitoring':    
        st.header("_Coming soon..._")    
        
    # ---------------------------------------------------------------------------------------------
    if view_button == 'Qualitative Assessment of Reliability':
        
        st.subheader("Qualitative assessment of reliability")
        qualitative_questionnaire()

    # ----------------------------------------------------------------------------------------------------
    if view_button == 'Business Relevance':
        
        st.subheader("Business Impact Evaluation")
        business_impact()
    
# ---------------------------------------------------------------------------------------------
if model_selector == 'Clustering':
    st.header("_Coming soon..._")   
# ---------------------------------------------------------------------------------------------
if model_selector == 'Gen AI':
    st.header("_Coming soon..._")   
    
# -------------Button-------------------------------------------------------------------------------------------------------
  
def HML(level):
    if level == "L":
        st.write("The business impact is low (L)")
    if level == "M":
        st.write("The business impact is Medium (M)")        
    if level == "H":
        st.write("The business impact is High (H)")      
    if level == "NA":
        st.write("The business impact is Not Applicable (NA)")          
        
# def summary_figure(df):
#     fig = px.line(df, x='x', y='y', markers=True,range_y=[-1,103]).update_layout(xaxis_title="Dimension", yaxis_title="Reliability Index"
#                                                                                  )

#     for i in range(len(df['y'])):
#         fig.add_annotation(x=i+0.02, y=df['y'][i]+3,text=str(df['y'][i])+"%",font_size=12, showarrow=False)

#     fig.add_hrect(y0=0, y1=50, line_width=2, fillcolor="red", opacity=0.5, annotation_text="Low", annotation_position="left",annotation_font_size=12)
#     fig.add_hrect(y0=50, y1=80, line_width=2, fillcolor="orange", opacity=0.5, annotation_text="Medium", annotation_position="left",annotation_font_size=12)
#     fig.add_hrect(y0=80, y1=100, line_width=2, fillcolor="green", opacity=0.5, annotation_text="High", annotation_position="left",annotation_font_size=12)
#     fig.update_layout(autosize=True,
#                       yaxis=dict(showgrid=False,showline=False,showticklabels=False,zeroline=True),
#                      paper_bgcolor= "rgba(0, 0, 0, 0)",plot_bgcolor= "rgba(0, 0, 0, 0)")
#     return fig
def summary_figure(df):
    # Create Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(11, 5))  
    # Plot the line chart
    ax.plot(df['x'], df['y'], marker='o')
    
    # Add annotations
    for i in range(len(df['y'])):
        ax.annotate(text=f"{df['y'][i]}%", xy=(df['x'][i], df['y'][i] + 3), fontsize=12)
    
    # Add colored rectangles
    ax.axhspan(0, 50, facecolor='red', alpha=0.5)
    ax.axhspan(50, 80, facecolor='orange', alpha=0.5)
    ax.axhspan(80, 100, facecolor='green', alpha=0.5)
    
    # Add labels "Low", "Medium", and "High"
    ax.text(-0.55, 25, 'Low', ha='center', fontsize=10)
    ax.text(-0.45, 65, 'Medium', ha='center', fontsize=10)
    ax.text(-0.55, 90, 'High', ha='center', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Reliability Index')
    ax.set_xlim(-0.7, len(df)-0.5)
    ax.set_ylim(-1, 103)
    ax.set_yticks([])
    
    # Hide grid and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Set background color
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    plt.tight_layout()  
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
# if 'stage' not in st.session_state:
#     st.session_state.stage = 0

# def set_state(i):
#     st.session_state.stage = i

# if st.session_state.stage == 0:
#     st.sidebar.button('Show Summary', on_click=set_state, args=[1])

# if st.session_state.stage >= 1:
#     df = pd.DataFrame(dict(
#         x = ['Data<br>Avalability','Data<br>Quality', 'Data<br>Rrepresentativeness',
#               'In-sample<br>Performance','Out-of-sample<br>Performance','Robustness<br>Score','Uncertainty<br>Score',
#               'Qualitative<br>Assessment'],
#         y = [st.session_state.Data_Availability, 
#               st.session_state.Data_Quality,
#               st.session_state.Data_Representativeness,
#               st.session_state.IRS_In,
#               st.session_state.IRS_Off,
#               st.session_state.Robutsness_Score,
#               st.session_state.Uncertainty_Score,
#               st.session_state.QS]
#         ))
#     df = df[df['y'].apply(lambda x: isinstance(x, (float, np.float64)))]
#     df['y'] = df['y'].astype(np.float64)
#     if len(df) == 0:
#         st.error('Please ensure that you have completed the assessment sections before proceeding to view the summary.')

#     elif len(df) > 0:
#         df['y'] = round(100*df['y'], 2)
#         fig = summary_figure(df)
#         st.pyplot(fig, use_container_width=True) 
    
#     st.sidebar.button('Show Summary', on_click=set_state, args=[0])
    


# ----------------------------------------------------------------------
page_names=['Show Summary','Calculate Score']
page = st.sidebar.radio('Navigation', page_names, index=None)

if page == 'Show Summary':
#if st.sidebar.button('Show summary', type='secondary', use_container_width=True):

    df = pd.DataFrame(dict(
        x = ['Data' + '\n' + 'Avalability','Data'+ '\n' +'Quality', 'Data'+ '\n' +'Rrepresentativeness',
              'In-sample'+ '\n' +'Performance','Out-of-sample'+ '\n' +'Performance','Robustness'+ '\n' +'Score',
             'Uncertainty'+ '\n' +'Score','Qualitative'+ '\n' +'Assessment'],
        y = [st.session_state.Data_Availability, 
              st.session_state.Data_Quality,
              st.session_state.Data_Representativeness,
              st.session_state.IRS_In,
              st.session_state.IRS_Off,
              st.session_state.Robutsness_Score,
              st.session_state.Uncertainty_Score,
              st.session_state.QS]
        ))
    df = df[df['y'].apply(lambda x: isinstance(x, (float, np.float64)))]
    df['y'] = df['y'].astype(np.float64)
    if len(df) == 0:
        st.error('Please ensure that you have completed the assessment sections before proceeding to view the summary.')

    elif len(df) > 0:
        df['y'] = round(100*df['y'], 2)
        fig = summary_figure(df)
        st.pyplot(fig, use_container_width=True)  
        st.session_state.figsummary = fig
        
if page == 'Calculate Score':
# if st.sidebar.button('Calculate score', type='secondary', use_container_width=True):
    weight_input = {}
    
    variables_dict = {"Data Availability": st.session_state.Data_Availability,
        "Data Quality": st.session_state.Data_Quality,
        "Data Representativeness": st.session_state.Data_Representativeness,
        "Model In-sample Performance":st.session_state.IRS_In,
        "Model Out-of-sample Performance": st.session_state.IRS_Off,
        "Model Robustness":st.session_state.Robutsness_Score,
        'Model Uncertainty':st.session_state.Uncertainty_Score,
        'Ongoing Monitoring':st.session_state.Ongoing_Monitoring}
    df=pd.DataFrame(list(variables_dict.items()), columns=["x", "y"])
    dimension_list = df['x'].to_list()
    agg_dimension = st.multiselect("Which dimensions do you want to include in final aggregation? ", dimension_list, 
                              help="Only selected dimensions will be considered in the final reliability index calculation.",
                              placeholder='Please select...') 
    
    if agg_dimension !=[]:
        df=df[df['x'].isin(agg_dimension)]
        incomplete_dimension = df[df['y'].apply(lambda x: not isinstance(x, (float, np.float64)))]['x'].to_list()
        if len(incomplete_dimension)>=1:
            st.error(f'The assessment for the following pre-specified diemsions has not been completed: {", ".join(incomplete_dimension)}. Please ensure that you have completed all the assessment sections thoroughly before proceeding to calculate final Reliability Index.')

        elif len(incomplete_dimension)==0:
            st.subheader("Final aggregation")
            st.markdown("Determine Weighting Method")
                
            response=st.selectbox("Do you have predetermined weights for each Reliability dimension that you would like us to use?", 
                                  ("Yes", "No"), index=None, placeholder="Please select...", 
                                  help='The weights will be used to aggregate individual reliability score of each dimension into final AI Reliability Confidence Index. The sum of all weights should equal 1.')
            
            if response == 'Yes':
                for dimension_name in df['x']:
                    weight_input[dimension_name] = st.number_input(f"Please enter the weight for {dimension_name}", 
                                                             min_value=0.00, max_value=1.00, step=0.0001,format="%.4f")
                    if weight_input is not None:
                        total_weight = sum(weight_input.values())
                        if round(total_weight,4) != 1.0:
                            st.error(f"The total of the weights you have entered is {round(total_weight,4)}. The sum of all weights should equal 1. Please adjust the weights accordingly.")
                        elif round(total_weight,4)==1.0:
                            st.session_state.weight_vector = np.array(list(weight_input.values()))
            elif response == 'No':
                Weight_AHP_data = st.file_uploader(label=":page_facing_up: Upload the Analytical Hierarchy Process (AHP) weight matrix file", type=["csv"],
                                                   help='The AHP Matrix is constructed through pairwise comparisons, where weights are assigned to each comparison using the following scale: 1 (equal importance), 3 (somewhat more important), 5 (definitely more important), 7 (much more important), and 9 (very much more important). This matrix indicates the relative importance of each dimension and will be used to calculate the weight for each dimension.')
                if Weight_AHP_data is not None:
                    Weight_AHP = pd.read_csv(Weight_AHP_data, index_col=0)
                    weight_AHP = Weight_AHP.to_numpy()
                    st.session_state.weight_vector = Final_Aggregation.Agg_weight(weight_AHP)
                    #st.write(st.session_state.weight_vector)
                    #st.write(st.session_state.weight_vector!=[])
                    
           # if st.session_state.weight_vector!=[]:
            if len(st.session_state.weight_vector)>0:
                st.session_state.score_vector = df['y'].to_list()
                Reliability = []
                Reliability_Index=Final_Aggregation.Reliability_Index(st.session_state.weight_vector, st.session_state.score_vector, st.session_state.QS)
                st.write(f"The Model Reliability Index is {round(Reliability_Index[0], 2)}%")
                st.write(f"The Model Reliability Index level is {Reliability_Index[1]}")
                Reliability.append(Reliability_Index[1])
                st.session_state.Indexs=Reliability_Index[1]

                selected_condition = Condition_number(st.session_state.Business, Reliability[0])
                # Find the index of the selected condition
                condition_index = conditions.index(selected_condition)
                # Load and display the corresponding image
                image = PILImage.open(f"{image_files_2[condition_index]}") # Adjust the file extension if necessary
                # st.image(image, caption=image_files[condition_index], use_column_width=True) 
                st.image(image, use_column_width=True) 
                st.session_state.reliability_index_image = image
                           
                # show weakness message
                if check_list(st.session_state.score_vector, 0.8):
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
                                unsafe_allow_html=True)
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


def compers(RS):
    if RS>0.8:
        rate="High"
    elif 0.5<=RS<0.8:
        rate="Medium"
    elif RS<0.5:
        rate="Low"
    else:
        rate="-"
    return rate

def businessF(Qs): 
    if Qs=="H":
        result="High"
    elif  Qs=="M":
        result="Medium"
    elif Qs=="L":
        result="Low"
    else:
        result="Not Applicable"
    return result
        
        
def export_answers_to_pdf(filename, questions, Q1):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    parts = []

    title_style = ParagraphStyle('Title', parent=styles['Title'], spaceBefore=-40)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14)
    text_style = styles['Normal']

    title = Paragraph("Reliability Confidence Index Report", title_style)
    parts.append(title)
    parts.append(Spacer(1, 15))
    
    ss="The reliability of AI system focuses on how consistently and accurately an AI model can produce the desired output or perform a specific task. "
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 5))
    ss="Due to the high-level complexity of the structure of AI models compared to traditional methods, the reliability becomes fundamental to ensure truth into those models and their applications within organizations.  "
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    ss="The reliability of AI models is evaluating using a semi-quantitative approach which combines both qualitative and quantitative measurement.  "
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 10))
    ss="Qualitative measurement: Use a questionnaire to evaluate whether model developers/users are following the best practices (according to AI model validation standard) throughout the model life cycle. "
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 5))
    ss=" Quantitative measurement: Use quantitative metrics to quantify different dimensions of reliability (i.e., data quality, performance evaluation, ongoing monitoring)."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    
    subtitle = Paragraph("<b>The data assessment</b>", subtitle_style)
    #paragraph2 = "The qualitative assessment of reliability"
    #paragraph2 = Paragraph(paragraph2, text_style)   
    parts.append(subtitle) 
    parts.append(Spacer(1, 15))
    
    ss="The data asssessment focuses on evaluating three (3) main dimensions: data availability, data representativeness and data quality assessment. The data quality component is evaluated based on various data quality dimensions (such as data coherence, data uniqueness, data completeness) depending on the business objective of the use case."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 5))
    ss="The following table provides us with high level summary of each data assessment dimension alongside with individual score and the rating (high/medium/low)."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    
    data2 = [['Dimension', 'Individual score (%)', 'Rating']]
    Availability=st.session_state.Data_Availability
    rates=compers(Availability)
    data2.append(['Data availability', round(100*Availability,2), rates])
    Data_Quality=st.session_state.Data_Quality
    rates=compers(Data_Quality)
    data2.append(['Data quality', round(100*Data_Quality,2), rates])
    Data_Representativeness=st.session_state.Data_Representativeness
    rates=compers(Data_Representativeness)
    data2.append(['Data representativeness', round(100*Data_Representativeness,2), rates])
    
    table = Table(data2)
    #df_Q2 = pd.DataFrame(list(data.items()), columns=["Questions", "Answers"])
    style = TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.gray),  # Background color for the header cell of the Dimension column
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Text color for the header cell of the Dimension column
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center alignment for Dimension column
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center alignment for Section column
        ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),  # Vertical alignment for Dimension column
        ('VALIGN', (1, 0), (1, -1), 'MIDDLE'),  # Vertical alignment for Section column
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Font for header row
        ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font size for all cells
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),  # Bottom padding for header row
        ('TOPPADDING', (0, 0), (-1, 0), 2),  # Top padding for header row
        ('LEFTPADDING', (0, 0), (-1, 0), 4),  # Left padding for header row
        ('RIGHTPADDING', (0, 0), (-1, 0), 4),  # Right padding for header row
        ('BACKGROUND', (1, 0), (-1, 0), colors.gray),  # Background color for header row
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines for all cells
 # Rotate text in Dimension column by 90 degrees
    ])
    
    # Apply the style to the table
    table.setStyle(style)
    parts.append(table)
    parts.append(Spacer(1, 15))
    

    subtitle = Paragraph("<b>The model assessment</b>", subtitle_style)
    #paragraph2 = "The qualitative assessment of reliability"
    #paragraph2 = Paragraph(paragraph2, text_style)   
    parts.append(subtitle) 
    parts.append(Spacer(1, 15))
    
    ss="The model assessment will focus on evaluating the model in sample and out of sample performance, the model robutsness checking and the uncertainty quantification."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 5))
    ss="The following table provides us with high level summary of each model assessment dimension alongside with individual score and the rating (high/medium/low."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    
    data3 = [['Dimension', 'Individual score (%)', 'Rating']]
    IRS_In=st.session_state.IRS_In
    rates=compers(IRS_In)
    data3.append(['In sample performance', round(100*IRS_In,2), rates])
    IRS_Off=st.session_state.IRS_Off
    rates=compers(IRS_Off)
    data3.append(['Out of sample performance', round(100*IRS_Off,2), rates]) 
    Robutsness_Score=st.session_state.Robutsness_Score
    rates=compers(Robutsness_Score)
    data3.append(['Robutsness checking', round(100*Robutsness_Score,2), rates])
    Uncertainty_Score=st.session_state.Uncertainty_Score
    rates=compers(Uncertainty_Score)
    data3.append(['Uncertainty quantification', round(100*Uncertainty_Score,2), rates])
    
    table1 = Table(data3)
    #df_Q2 = pd.DataFrame(list(data.items()), columns=["Questions", "Answers"])
    style = TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.gray),  # Background color for the header cell of the Dimension column
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Text color for the header cell of the Dimension column
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center alignment for Dimension column
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center alignment for Section column
        ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),  # Vertical alignment for Dimension column
        ('VALIGN', (1, 0), (1, -1), 'MIDDLE'),  # Vertical alignment for Section column
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Font for header row
        ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font size for all cells
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),  # Bottom padding for header row
        ('TOPPADDING', (0, 0), (-1, 0), 2),  # Top padding for header row
        ('LEFTPADDING', (0, 0), (-1, 0), 4),  # Left padding for header row
        ('RIGHTPADDING', (0, 0), (-1, 0), 4),  # Right padding for header row
        ('BACKGROUND', (1, 0), (-1, 0), colors.gray),  # Background color for header row
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines for all cells
 # Rotate text in Dimension column by 90 degrees
    ])
    
    # Apply the style to the table
    table1.setStyle(style)
    parts.append(table1)
    parts.append(Spacer(1, 15))
 
    parts.append(PageBreak())
    
    
    subtitle = Paragraph("<b>The qualitative assessment of reliability</b>", subtitle_style)
    #paragraph2 = "The qualitative assessment of reliability"
    #paragraph2 = Paragraph(paragraph2, text_style)   
    parts.append(subtitle) 
    parts.append(Spacer(1, 15))
    
    ss="The qualitative assessment of reliability consists of using a Yes/No/NA qualitative questionnaire to assess whether the model developers/users follow best practices during the model development and using phases."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    
    datas = [['Questions', 'Answers']]
    #data=st.session_state.questionnaire1
    user_input=st.session_state.user
    for question in Q1:
        datas.append([question, user_input[question]])
    
    table = Table(datas)
    
    #df_Q2 = pd.DataFrame(list(data.items()), columns=["Questions", "Answers"])
    style = TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.gray),  # Background color for the header cell of the Dimension column
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Text color for the header cell of the Dimension column
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center alignment for Dimension column
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center alignment for Section column
        ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),  # Vertical alignment for Dimension column
        ('VALIGN', (1, 0), (1, -1), 'MIDDLE'),  # Vertical alignment for Section column
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Font for header row
        ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font size for all cells
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),  # Bottom padding for header row
        ('TOPPADDING', (0, 0), (-1, 0), 2),  # Top padding for header row
        ('LEFTPADDING', (0, 0), (-1, 0), 4),  # Left padding for header row
        ('RIGHTPADDING', (0, 0), (-1, 0), 4),  # Right padding for header row
        ('BACKGROUND', (1, 0), (-1, 0), colors.gray),  # Background color for header row
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines for all cells
 # Rotate text in Dimension column by 90 degrees
    ])
    
    # Apply the style to the table
    table.setStyle(style)
    parts.append(table)
    parts.append(Spacer(1, 15))
    
    #<b>The model assessment</b>
    Qs=st.session_state.QS 
    ss=f" The qualitative reliability score is <b> {round(100 * Qs, 2)}% </b>"
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    parts.append(PageBreak())

    subtitle = Paragraph("<b>The business impact evaluation</b>", subtitle_style)
    #paragraph2 = "The qualitative assessment of reliability"
    #paragraph2 = Paragraph(paragraph2, text_style)   
    parts.append(subtitle) 
    parts.append(Spacer(1, 15))
    
    ss="The business impact evaluation examines the influence of your AI solution across four key areas: revenue generation, AI compliance, reputation management, and regulatory enablement."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    #parts.append(Spacer(1, 2))
    ss="This evaluation is performed using a qualitative questionnaire which is summarized in the following table."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    
    data1 = [['Questions', 'Answers']]
    user_input1=st.session_state.user1
    critical=st.session_state.critical
    sa="Is the AI use cases involved in life-critical or safety-critical operations?"
    # sa = Paragraph(sa, text_style)  
    data1.append([sa, critical])
    
    for question in questions:
        data1.append([question, user_input1[question]])
    
    table1 = Table(data1)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.gray),  # Background color for the header cell of the Dimension column
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Text color for the header cell of the Dimension column
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center alignment for Dimension column
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center alignment for Section column
        ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),  # Vertical alignment for Dimension column
        ('VALIGN', (1, 0), (1, -1), 'MIDDLE'),  # Vertical alignment for Section column
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Font for header row
        ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font size for all cells
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),  # Bottom padding for header row
        ('TOPPADDING', (0, 0), (-1, 0), 2),  # Top padding for header row
        ('LEFTPADDING', (0, 0), (-1, 0), 4),  # Left padding for header row
        ('RIGHTPADDING', (0, 0), (-1, 0), 4),  # Right padding for header row
        ('BACKGROUND', (1, 0), (-1, 0), colors.gray),  # Background color for header row
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines for all cells
 # Rotate text in Dimension column by 90 degrees
    ])
    
    # Apply the style to the table
    table1.setStyle(style)
    parts.append(table1)
    parts.append(Spacer(1, 15))
    
    Qs=st.session_state.Business
    rate=businessF(Qs)
    ss=f" The business impact is rating as <b> {rate} </b>"
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    #parts.append(PageBreak())
    
    subtitle = Paragraph("<b>The final aggregation</b>", subtitle_style)
    #paragraph2 = "The qualitative assessment of reliability"
    #paragraph2 = Paragraph(paragraph2, text_style)   
    parts.append(subtitle) 
    parts.append(Spacer(1, 15))
    
    ss="The following plot provides a summary of individual scores across the reliability dimensions including the qualitative component. It provides us with a view on which reilability dimension has a high/medium/low reliability score."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    buf = BytesIO()
    st.session_state.figsummary.savefig(buf, format='png')
    buf.seek(0)
    img = Image(buf, width=400, height=250)

    parts.append(img)
    parts.append(Spacer(1, 5))
    

    ss="After aggregating all the individual score into a single reliability score ranking between 0 and 1 and then convert this final score into High/Medium/Low basis, provide the following graph that shows the reliability index alongside the business impact of the use case."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    buf = BytesIO()
    st.session_state.reliability_index_image.save(buf, format='png')
    buf.seek(0)
    img = Image(buf, width=400, height=250)
    parts.append(img)
    parts.append(Spacer(1, 15))
    
    Qs=st.session_state.Business
    result=businessF(Qs)
    indexs=st.session_state.Indexs
    ss=f"The use case has been assigned a <b>{indexs}</b> reiability confidence level and a <b>{result}</b> business impact rating."
    ss = Paragraph(ss, text_style)  
    parts.append(ss) 
    parts.append(Spacer(1, 15))
    
    #<b>The business impact evaluation</b>
    
    if check_list(st.session_state.score_vector, 0.8) | (abs(st.session_state.IRS_In - st.session_state.IRS_Off)>0.1):
        ss="Based on the quantitative and qulitative assessment results, the table below summaeizes the current challenges and suggestions for improvement."
        ss = Paragraph(ss, text_style)  
        parts.append(ss) 
        parts.append(Spacer(1, 15))
        
        data4 = [['Challenges', 'Recommendations']]
        if (st.session_state.Data_Availability<0.8)|(st.session_state.Data_Quality<0.8)|(st.session_state.Data_Representativeness<0.8):
            data4.append(['Ineffective data governance', 'Requires a mitigation plan to improve data availability, data quality, and data representativeness.'])
        if (st.session_state.IRS_In<0.8)|(st.session_state.IRS_Off<0.8)|(st.session_state.Robutsness_Score<0.8)|(st.session_state.Uncertainty_Score<0.8):
            data4.append(['Poor model performance', 'Requires model recalibration or retraining to improve model effectiveness.'])
        if abs(st.session_state.IRS_In - st.session_state.IRS_Off)>0.1:
            data4.append(['Over/under fitting', 'Requires a remediation strategy and careful tuning of model complexity and hyperparameters to build robust models.'])
        
        table = Table(data4)
        #df_Q2 = pd.DataFrame(list(data.items()), columns=["Questions", "Answers"])
        style = TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.gray),  # Background color for the header cell of the Dimension column
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Text color for the header cell of the Dimension column
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center alignment for Dimension column
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center alignment for Section column
            ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),  # Vertical alignment for Dimension column
            ('VALIGN', (1, 0), (1, -1), 'MIDDLE'),  # Vertical alignment for Section column
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Font for header row
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font size for all cells
            ('BOTTOMPADDING', (0, 0), (-1, 0), 2),  # Bottom padding for header row
            ('TOPPADDING', (0, 0), (-1, 0), 2),  # Top padding for header row
            ('LEFTPADDING', (0, 0), (-1, 0), 4),  # Left padding for header row
            ('RIGHTPADDING', (0, 0), (-1, 0), 4),  # Right padding for header row
            ('BACKGROUND', (1, 0), (-1, 0), colors.gray),  # Background color for header row
            ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines for all cells
     # Rotate text in Dimension column by 90 degrees
        ])
        
        # Apply the style to the table
        table.setStyle(style)
        parts.append(table)
        parts.append(Spacer(1, 15))
    
    doc.build(parts)
    st.success("Answers exported to PDF successfully!")
    


if st.sidebar.button('Export to PDF'):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_filename = f"{current_folder}/Reliability_Index_Report.pdf"
    
    questions = ["Q1. To what extent do the model’s output and its associated errors impact regulatory compliance?", 
                 "Q2. To what extent do the model’s output and its errors impact financial, or significant business decisions?",
                 "Q3. Is the AI in a public-facing role, potentially influencing public opinion or behaviors?", 
                 "Q4. To what extent the model output and its associated errors impact the organization’s reputation?"]
    # Create a dictionary to store user input
    
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
    export_answers_to_pdf(output_filename, questions, Q1)
    



# Define the reassessment function to toggle the reassessment state
def reassessment_summary():
    # # st.session_state.reassessment_active = False
    # st.session_state.reassessment_active = not st.session_state.reassessment_active
    st.sidebar.markdown("***")
    st.subheader("Reassessment")
    st.markdown('Please navigate to **Upload File** page to submit new files for reassessment.')
    st.session_state.clear()


if "reassessment_active" not in st.session_state:
    st.session_state.reassessment_active = False
        
# if st.sidebar.button('Reassessment', type='secondary', use_container_width=True):
if page == 'None':
    st.write("Please select an option from the Navigation sidebar.")


if st.sidebar.button('Reassessment'):
    st.session_state.reassessment_active = not st.session_state.reassessment_active

    reassessment_summary()












