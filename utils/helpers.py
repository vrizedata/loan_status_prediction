import pandas as pd

def clean_cat_cols(df):
    # CAT COLS

    # job title 
    jobs = [ 'Teacher', 'Manager', 'Registered Nurse', 'RN', 'Supervisor', 'Sales', 'Project Manager', 'Owner', 
        'Driver', 'Office Manager', 'manager', 'Director', 'General Manager', 'Engineer', 'teacher']
    # filter dataframe as per above condition
    df['emp_title'] = df['emp_title'].apply(lambda x: 'Other' if x not in jobs else x)

    # issue d
    df['issue_d'] = df['issue_d'].apply(lambda x: x.split('-')[0])


    # title - taking top 23 titles (99 percentile)
    titles = ['Debt consolidation', 'Credit card refinancing', 'Home improvement', 'Other', 'Debt Consolidation', 'Major purchase',
          'Consolidation', 'debt consolidation', 'Business', 'Medical expenses', 'Car financing', 'Moving and relocation', 
          'Vacation', 'Credit Card Consolidation', 'consolidation', 'Debt Consolidation Loan', 'Home buying', 
          'Consolidation Loan', 'Personal Loan', 'Credit Card Refinance', 'Home Improvement', 'Credit Card Payoff', 
          'Consolidate']


    df['title'] = df['title'].apply(lambda x: 'Other' if x not in titles else x)

    # earliest cr line
    df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: x.split('-')[0])
    
    return df


def one_hot_cat_cols(df,cat_cols,cat_dict):
    results = []
    for col_name in cat_cols:
        known_cats = cat_dict[col_name]
        df_cat = pd.Categorical(df[col_name].values, categories = known_cats)
        df_cat = pd.get_dummies(df_cat,prefix=col_name)
        results.append(df_cat)
    
    
    df_cat = pd.concat(results,axis=1).reset_index(drop=True)
    return df_cat
          
def clean_num_cols(df):
    # pub rec -> [1,2,3,4,5,6(>5)]
    df['open_acc']  = df['open_acc'].apply(lambda x: 38 if x>=38 else x)
    df['revol_bal']  = df['revol_bal'].apply(lambda x: 85757 if x>=85757 else x)
    df['revol_util'] = df['revol_util'].apply(lambda x: 102.5 if x>=102.5 else x)
    df['total_acc'] = df['total_acc'].apply(lambda x: 78 if x>=78 else x)
    #mort_acc -> [1,2.....13,14(>13)]
    return df


def std_num_cols(df,num_cols,scalar):
    print(" columns type-------------------")
    for i in num_cols:
        print(type(i))
       
    df[num_cols] = scalar.transform(df[num_cols])
    return df[num_cols]
    
    
def feature_engineer(df,cat_dict,std_scalar):    
    
    df = df.drop(columns=['address'])
    #CAT COLS
    cat_cols = ['term','grade','sub_grade','emp_title','emp_length','home_ownership','verification_status',
        'issue_d','purpose','title','earliest_cr_line','initial_list_status',
        'application_type']

    df_cat = clean_cat_cols(df[cat_cols])
    df_cat = one_hot_cat_cols(df_cat,cat_cols,cat_dict)


    # NUM COLS
    num_cols =['loan_amnt','int_rate','installment','annual_inc','dti','open_acc','pub_rec',
           'revol_bal','revol_util','total_acc','mort_acc','pub_rec_bankruptcies']

    df_num = std_num_cols(df,num_cols,std_scalar)
    X = pd.concat([df_num,df_cat],axis=1)
    return X