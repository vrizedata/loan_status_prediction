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

#columns = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_title','emp_length','home_ownership','annual_inc','verification_status','issue_d','purpose','title','dti','earliest_cr_line','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','application_type','mort_acc','pub_rec_bankruptcies','address']

#print("-------columns_length ",len(columns))

# normal_columns = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership','annual_inc','verification_status','purpose','dti','open_acc','revol_bal','revol_util','total_acc','initial_list_status','application_type','pub_rec_bankruptcies','address']
# print("-------columns_length ",len(columns))

inp_dict = {}
#categ_cols = ['term','grade','sub_grade','emp_title','emp_length','home_ownership','verification_status','issue_d','purpose','title','earliest_cr_line','initial_list_status','application_type','address']

title_options = ['Debt consolidation', 'Credit card refinancing', 'Home improvement', 'Other', 'Debt Consolidation', 'Major purchase',
'Consolidation', 'debt consolidation', 'Business', 'Medical expenses', 'Car financing', 'Moving and relocation',
'Vacation', 'Credit Card Consolidation', 'consolidation', 'Debt Consolidation Loan', 'Home buying',
'Consolidation Loan', 'Personal Loan', 'Credit Card Refinance', 'Home Improvement', 'Credit Card Payoff',
'Consolidate','Other']

emp_title_options = [ 'Teacher', 'Manager', 'Registered Nurse', 'Other','RN', 'Supervisor', 'Sales', 'Project Manager', 'Owner',
'Driver', 'Office Manager', 'manager', 'Director', 'General Manager', 'Engineer', 'teacher']

month_options = ['Oct', 'Sep', 'Aug', 'Nov', 'Dec', 'Jul', 'Mar', 'Jan', 'Jun', 'May', 'Apr', 'Feb']

home_ownership_options = ['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'NONE', 'ANY']

grade_options = ['A','B','C','D','E' ,'F','G']

sub_grade_options = ['1','2','3','4','5']

verification_status_options = ['Verified','Source Verified','Not Verified']

purpose_options = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business', 'car', 'medical', 'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'educational']

initial_list_status_options = ['f', 'w']         

application_type_options = ['INDIVIDUAL', 'JOINT', 'DIRECT_PAY']

term_options = ['36 months', '60 months']

emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years' ,'10+ years']
      
#print(len(categ_cols))

with st.form("my_form"):

    inp_dict['loan_amnt'] = st.number_input('loan_amnt (The listed amount of the loan applied for by the borrower)')

    inp_dict['term'] = st.selectbox('term (The number of payments on the loan', term_options)

    inp_dict['int_rate'] = st.number_input('int_rate (Interest rate on the loan)')

    inp_dict['installment'] = st.number_input('installment (The monthly payment owed by the borrower if the loan originates)')

    inp_dict['grade'] = st.selectbox("grade (Loan grade)", grade_options)

    inp_dict['sub_grade'] = st.selectbox("sub_grade (Loan sub grade)", sub_grade_options)

    inp_dict['emp_title']=st.selectbox('emp_title (Job title of borrower when applying for loan)', emp_title_options)

    inp_dict['emp_length'] = st.selectbox('emp_length (Employment length of borrower in years at time of applying for loan)', emp_length_options)

    inp_dict['home_ownership'] = st.selectbox("home_ownership (The home ownership status provided by the borrower during registration or obtained from the credit report)", home_ownership_options)

    inp_dict['annual_inc'] = st.number_input('annual_inc (The self-reported annual income provided by the borrower during registration)')    

    inp_dict['verification_status'] = st.selectbox("verification_status (wheather the income was verified by Lending club)", verification_status_options)

    inp_dict['issue_d'] = st.selectbox("issue_d (Month in which loan was funded)", month_options)

    inp_dict['purpose'] = st.selectbox("purpose (pupose of loan provided by borrower)", purpose_options)

    inp_dict['title'] = st.selectbox('title (The loan title provided by the borrower)', title_options)

    inp_dict['address'] = st.text_input("address (The state provided by the borrower in the loan application)")

    inp_dict['dti'] = st.number_input('dti (A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income)')

    inp_dict['earliest_cr_line']=st.selectbox("earliest_cr_line (Month in which earliest reported credit line was opened)",month_options)
    
    inp_dict['open_acc'] = st.number_input('open_acc (Number of open credit lines in borrowers credit file)')

    inp_dict['pub_rec'] = st.number_input('pub_rec (Number of derogatory public records)')

    inp_dict['revol_bal'] = st.number_input('revol_bal (Total credit revolving balance)')

    inp_dict['revol_util'] = st.number_input('revol_util (Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit)')

    inp_dict['total_acc'] = st.number_input("total_acc (The total number of credit lines currently in the borrower's credit file)")

    inp_dict['initial_list_status']=st.selectbox('initial_list_status (The initial listing status of the loan)', initial_list_status_options)

    inp_dict['application_type']=st.selectbox('application_type (Type of loan application)', application_type_options)

    inp_dict['mort_acc']=st.number_input('mort_acc (Number of mortgage accounts)')

    inp_dict['pub_rec_bankruptcies']=st.number_input('pub_rec_bankruptcies (Number of public record bankruptcies)')


    
    inp_dict['sub_grade'] = inp_dict['grade']+inp_dict['sub_grade']
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

        #loaded_model = mlflow.pyfunc.load_model(model_uri='artifacts/4/'+run_id+'/artifacts/model')

        loaded_model = mlflow.sklearn.load_model(model_uri='artifacts/4/'+run_id+'/artifacts/model')


        # Use the abstract function in FastTextWrapper to fetch the trained model.
        print("model type -------------------",type(loaded_model))
        ###############################################
        # input_list = [[18000,'36 months',5.32,542.07,'A','A1','Software Development Engineer','2 years	MORTGAGE',125000,'Source Verified','Sep-15','Fully Paid','home_improvement','Home improvement',1.36,'Aug-05',8,0,4178,4.9,25,'f','INDIVIDUAL',3,0,'1008 Erika Vista Suite 748 East Stephanie, TX 22690']]
        # print("-------------input_lst- ",len(input_list))


        # columns = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_title','emp_length','home_ownership','annual_inc','verification_status','issue_d','purpose','title','dti','earliest_cr_line','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','application_type','mort_acc','pub_rec_bankruptcies','address']
        # print("-------------input_cols- ",len(columns))
        # input_df = pd.DataFrame(input_list,columns=columns)
        # input_df.to_csv("inp_df_m.csv")
        # print(input_df.shape)

        # print("-------------input_df- ",len(input_df))
        #####################################################
        # x_point = df_main.iloc[10].to_json()
        # with open("inp_features.json", "w") as outfile:
        #     outfile.write(x_point)
        # print(x_point)
        # # create dataframe from data
        # data = json.loads(x_point)
        # input_df = pd.DataFrame.from_dict(data,orient='index').T
        input_df = pd.DataFrame.from_dict(inp_dict,orient='index').T
        #label = input_df['loan_status'].values[0]

        #input_df.to_csv("inp_df_org.csv")
        #
        input_features= feature_engineer(input_df,cat_dict,std_scalar)

        print("-------------input_features- ",len(input_features))


        out = loaded_model.predict(input_features)[0]
        prob = loaded_model.predict_proba(input_features)

        print('')
        #print(f'Predicted_class: {out}')
        #print(f'Actual label:{label}')
        print(f'Predicted probability: {prob[0][0]}')
        print(f'Predicted label: {label_map[out]}')

        st.write(f'Predicted probability: {prob[0][0]}')
        st.write(f'Predicted label: {label_map[out]}')




