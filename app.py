import streamlit as st
import mlflow
import mlflow
import pandas as pd
import numpy as np
import datetime
import os 
from from_root import from_root
import pickle
import joblib
import json
from utils.helpers import clean_cat_cols, one_hot_cat_cols, clean_num_cols, std_num_cols, feature_engineer


st.title("Loan status prediction")

# columns =  ['term','grade','sub_grade','emp_title','emp_length','home_ownership','verification_status',
#  'issue_d','purpose','title','earliest_cr_line','initial_list_status',
#  'application_type','address']

columns = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_title','emp_length','home_ownership','annual_inc','verification_status','issue_d','purpose','title','dti','earliest_cr_line','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','application_type','mort_acc','pub_rec_bankruptcies','address']

print("-------columns_length ",len(columns))

# normal_columns = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership','annual_inc','verification_status','purpose','dti','open_acc','revol_bal','revol_util','total_acc','initial_list_status','application_type','pub_rec_bankruptcies','address']
# print("-------columns_length ",len(columns))

inp_dict = {}
categ_cols = ['term','grade','sub_grade','emp_title','emp_length','home_ownership','verification_status','issue_d','purpose','title','earliest_cr_line','initial_list_status','application_type','address']

title_options = ['Debt consolidation', 'Credit card refinancing', 'Home improvement', 'Other', 'Debt Consolidation', 'Major purchase',
'Consolidation', 'debt consolidation', 'Business', 'Medical expenses', 'Car financing', 'Moving and relocation',
'Vacation', 'Credit Card Consolidation', 'consolidation', 'Debt Consolidation Loan', 'Home buying',
'Consolidation Loan', 'Personal Loan', 'Credit Card Refinance', 'Home Improvement', 'Credit Card Payoff',
'Consolidate','Other']

emp_title_options = [ 'Teacher', 'Manager', 'Registered Nurse', 'Other','RN', 'Supervisor', 'Sales', 'Project Manager', 'Owner',
'Driver', 'Office Manager', 'manager', 'Director', 'General Manager', 'Engineer', 'teacher']

pub_rec_options =  [1,2,3,4,5,6]
mort_acc_options = [1,2,3,4,5,6,7,8,9,10,11,12,13]

month_options = ['Oct', 'Sep', 'Aug', 'Nov', 'Dec', 'Jul', 'Mar', 'Jan', 'Jun', 'May', 'Apr', 'Feb']

select_box_columns = ['title','emp_title','pub_rec','mort_acc','earliest_cr_line','issue_d']

home_ownership_options = ['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'NONE', 'ANY']

print(len(categ_cols))

with st.form("my_form"):
    for col in columns:

        if col in select_box_columns:
            if col == 'title':
                inp_dict[col]=st.selectbox('What is your job title?', title_options)
            if col == 'emp_title':
                inp_dict[col]=st.selectbox('Select loan title', emp_title_options)
            if col == 'pub_rec':
                inp_dict[col]=st.selectbox('Number of derogatory public records', pub_rec_options)
            if col == 'mort_acc':
                inp_dict[col]=st.selectbox('Number of mortgage accounts', mort_acc_options)
            if col == 'earliest_cr_line':
                inp_dict[col]=st.selectbox("Month in which earliest reported credit line was opened",month_options)
            if col == 'issue_d':
                inp_dict[col]=st.selectbox("select the month in which loan was funded",month_options)
            # if col == 'home_ownership':
            #     inp_dict[col] = st.selectbox("select the type of home ownership", home_ownership_options)
            # if col== 'grade':
            #     inp_dict[col] = st.selectbox("select loan grade",['A','B','C','D',''])
            
        else:
            
            if col not in categ_cols:
                inp_dict[col] = st.number_input(col)
            else:
                inp_dict[col] = st.text_input(col)
        
    inp_dict["earliest_cr_line"] = inp_dict["earliest_cr_line"]+"_2020"
    inp_dict["issue_d"] = inp_dict["issue_d"]+"_2020"

    # Every form must have a submit button.
    submitted = st.form_submit_button("Predict loan status")
    if submitted:
        #st.write("Submitted")
        #st.write("slider", slider_val, "checkbox", checkbox_val)
        print(inp_dict)
        print(type(inp_dict['int_rate']))




        #loading cat cols
        with open(os.path.join(from_root(),'artifacts','prepare_data','cat_cols.pickle'), 'rb') as handle:
            cat_dict = pickle.load(handle)
        #loading std scalar
        with open(os.path.join(from_root(),'artifacts','prepare_data','std_scaler.bin'),'rb')as handle:
            std_scalar = joblib.load(handle)
            
        label_map = {0:'Fully Paid',1:'Charged Off'}

        run_id = 'ecc09ad03c694525bd58b2128e55f874'

        loaded_model = mlflow.pyfunc.load_model(model_uri='artifacts/4/'+run_id+'/artifacts/model')


        print("model type -------------------",type(loaded_model))
        input_df = pd.DataFrame.from_dict(inp_dict,orient='index').T
        input_features= feature_engineer(input_df,cat_dict,std_scalar)

        print("-------------input_features- ",len(input_features))
        out = loaded_model.predict(input_features)[0]
        print(f'Predicted label: {label_map[out]}')

        st.title(f'Predicted label: {label_map[out]}')



