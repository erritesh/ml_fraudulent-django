from tkinter import E
from tokenize import String
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.http import HttpResponse
#Import necessary dataset
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import _tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score #Import scikit-learn metrics module for accuracy calculation
from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table
from django_pandas.io import read_frame
import mysql.connector
from mysql.connector import Error
import MySQLdb
from  sqlalchemy import create_engine
import pandas


def evaluateViews(request):
    print("Machine Learning Evaluation Started...")
    applicant_details = Applicant_Details.objects.all().values('app_email', 'app_id', 'app_mailing', 'app_onphone', 'app_ssn', 'app_start_time', 'app_submission_time', 'applicant_name', 'classification', 'geoLocation', 'origin_ip', 'renter', 'requested_amount','unit_type')
    #read the df from DB Applicant_Details
    df = read_frame(applicant_details)
    #print(df)
    

    #format the data (time interval)
    df['app_start_time'] = pd.to_datetime(df['app_start_time'])
    df['app_submission_time'] = pd.to_datetime(df['app_submission_time'])
    #calculate the time difference
    df['Application_Duration']  = (df['app_submission_time'] - df['app_start_time']) / np.timedelta64(1,'s')
    
    
    #addition table for the # of applications sending from each applicants
    app_count = df.groupby('app_ssn').size().reset_index() 
    app_count.columns =['app_ssn', 'Application_Count_From_Applicant']
    #merge back to the orginal df table
    df = df.merge(app_count, on = 'app_ssn', how = 'left')
    #setting up training dataset
    df_training = df.dropna(subset=['classification'])
    feature_cols = ['Application_Duration', 'requested_amount', 'geoLocation', 'Application_Count_From_Applicant']
    X_training = df_training[feature_cols] # Features

   
    Y_training = pd.DataFrame(df_training['classification'])# Target variable
    X_all = df[feature_cols]
   
    # transfer categorical data to dummy variable for both dataset
    x_training_dummy= pd.get_dummies(X_training,drop_first=True)
    x_all_dummy = pd.get_dummies(X_all,drop_first=True)
    
    #x_all_dummy.to_csv('sample_xall_local.csv')
    #prevent missing columns from all dataset and x traiing
    missing_cols = set(x_all_dummy.columns) - set(x_training_dummy.columns )
    for c in missing_cols:
        x_training_dummy[c] = 0

    x_training_dummy = x_training_dummy[x_training_dummy.columns]
    #print(x_training_dummy)

    #re-name the columns to be more readable later
    x_training_dummy.columns = x_training_dummy.columns.str.replace('_', ' ')
    x_all_dummy.columns = x_all_dummy.columns.str.replace('_', ' ')
    #print(x_all_dummy.columns)
    #cross validation from the max depth and min sample leaf
    parameters = {'max_depth':range(1,20),'min_samples_leaf':range(1,20)}
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
    clf.fit(X=x_training_dummy, y=Y_training)
    tree_model = clf.best_estimator_
    #Predict the response for the whole dataset
    y_pred_class =tree_model.predict(x_all_dummy)
    y_prob_prob = pd.DataFrame(tree_model.predict_proba(x_all_dummy)).iloc[:, 0]
    print(y_pred_class)
    print(y_prob_prob)
    print(x_all_dummy)
    #getting the decision critiria

    n_nodes = tree_model.tree_.node_count
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right
    feature = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    #print (n_nodes)
    #print (children_left)
    #print(children_right)
    #print(feature)
    #print(threshold)

    def find_path(node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (children_left[node_numb] !=-1):
            left = find_path(children_left[node_numb], path, x)
        if (children_right[node_numb] !=-1):
            right = find_path(children_right[node_numb], path, x)
        if left or right :
            return True
        path.remove(node_numb)
        return False

    def get_rule(path, column_names):
        mask = ''
        for index, node in enumerate(path):
            #We check if we are not in the leaf
            if index!=len(path)-1:
                # Do we go under or over the threshold ?
                if (children_left[node] == path[index+1]):
                    mask += "{} <= {}\t".format(column_names[feature[node]], threshold[node])
                else:
                    mask += "{} > {}\t".format(column_names[feature[node]], threshold[node])
        # We insert the & at the right places
        mask = mask.replace( "\t",", ", mask.count("\t") - 1)
        mask = mask.replace("\t", "")
        return mask
    #apply the leave id for each record on the dataset
    leave_id = tree_model.apply(x_all_dummy)
    paths ={}
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(0, path_leaf, leaf)
        paths[leaf] = np.unique(np.sort(path_leaf))

    rules = {}
    for key in paths:
        rules[key] = get_rule(paths[key], x_all_dummy.columns)
       
    # leave id for each record
    leave_id_df = pd.DataFrame(leave_id)
    
    leave_id_df.columns = ['leave_id']
    # setting up the final output data
    result = pd.DataFrame()
    
    result['app_id'] = df['app_id']
    result['classification'] = df['classification']
    result['predict_class'] =  y_pred_class
    result['Risk_Score'] =  round(y_prob_prob*100,1)
    result['Decision_Criteria'] = leave_id_df['leave_id'].map(rules)
    #print (result)

    # connect to DB 
    con = MySQLdb.connect('localhost','root','Admin', 'ml_fraud_detection')
    my_sqlconn= create_engine("mysql+mysqldb://root:Admin@localhost/ml_fraud_detection")
    df = pd.DataFrame(data=result)
    df.to_sql(con=my_sqlconn,name='ml_fraudulent_app_risk_table',if_exists='replace',index=False)
    print("Risk table has been updated successfully !")

    #importance rank calculation
    fraud_result = result[result['predict_class'] == 'Fraud']
    total = fraud_result.shape[0]
    criteria_count = pd.DataFrame(fraud_result['Decision_Criteria'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('Decision_Criteria'))
    affected = pd.DataFrame(criteria_count.groupby('Decision_Criteria').size()).reset_index()
    affected.columns = ('Decision_Criteria','Count')
    affected['Importance'] = (affected['Count']/total*100).astype(int)
    importance_rank = affected.drop(columns=['Count']).sort_values('Importance',ascending= False).reset_index(drop=True)
    print("========== Importance rank =================")
    print(importance_rank)

    
    #return HttpResponseRedirect(request.GET.get('next'))

    #return redirect('')
    return HttpResponse("ML Script Runned successfully")
    #return render(request, "ml_fraudulent_app/index.html")
