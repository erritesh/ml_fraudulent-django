from gettext import find
from django.http.response import HttpResponse
from django.shortcuts import redirect, render
from django.http import HttpResponse
from numpy import int64
from sqlalchemy import null
from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table
from django.http import JsonResponse
from django.views.generic import View
from django.db import connection
from django.db.models.functions import ExtractMonth
from django.db.models import Count
from django_pandas.io import read_frame
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go 
#from plotly.offline import iplot
import json
import plotly
from django.contrib import messages
import mysql.connector
#from decimal import Decimal
from csv import reader
from urllib import response
import geoip2.database
from pprint import pprint
import mysql.connector
from mysql.connector import Error
import logging
import logging.config
logger = logging.getLogger('django')
from rest_framework.decorators import api_view
from django.core.paginator import Paginator,EmptyPage, PageNotAnInteger
import simplejson
from django.http import Http404
from django.utils import timezone
import pytz
from django.db.models import Subquery, OuterRef
from sqlalchemy import create_engine
import mysql.connector
import pandas as pd
import numpy as np
from mysql.connector import errorcode
import MySQLdb
# Import necessary dataset
import pandas as pd
import numpy as np
from django.http import HttpResponse
import re
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import _tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score




# Create your views here.
def login(request):
    return render(request,"ml_fraudulent_app/login.html")
    
def home(request):
    total_query_set = Applicant_Details.objects.all().count()
    low_queryset = Risk_Table.objects.filter(classification=None) & Risk_Table.objects.filter(Risk_Score__range=(0,25))
    medium_queryset = Risk_Table.objects.filter(classification=None) & Risk_Table.objects.filter(Risk_Score__range=(26,50))
    high_queryset = Risk_Table.objects.filter(classification=None) & Risk_Table.objects.filter(Risk_Score__range=(51,75))
    critical_queryset = Risk_Table.objects.filter(classification=None) & Risk_Table.objects.filter(Risk_Score__range=(77,100))

    lowCount=low_queryset.count()   
    mediumCount = medium_queryset.count()
    highCount = high_queryset.count()
    criticalCount = critical_queryset.count()
      
    #applicant_details = Applicant_Details.objects.get_queryset().order_by('app_id')

    applicant_details = Applicant_Details.objects.all().values('app_email', 'app_id', 'app_mailing', 'app_onphone', 'app_ssn', 'app_start_time', 'app_submission_time', 'applicant_name', 'classification', 'geoLocation', 'origin_ip', 'renter', 'requested_amount', 'unit_type','risk_table__Decision_Criteria','risk_table__predict_class','risk_table__Risk_Score').order_by('app_id')

    # df = read_frame(applicant_details)
    # json_records = df.reset_index().to_json(orient ='records')
    # data = []
    # data = json.loads(json_records)
    
    #template_to_see = render_template('updatecustomer.html', customers = customers)
    pagesno = request.GET.get('selectedpage')

    finalpage = int(pagesno or 10)
    
    #print(pagesno)

    page = request.GET.get('page', 1)
    paginator = Paginator(applicant_details, 10)

    try:
        applicant_list = paginator.page(page)
    except PageNotAnInteger:
        applicant_list = paginator.page(1)
    except EmptyPage:
       applicant_list = paginator.page(paginator.num_pages)

    #logger.info (applicant_details)
    #dict= {[{user_id: {}},{userid2:{}}]}
    #newvar= {user_id: risk_score}
    

    context = { 
                "total_query_set":total_query_set,
                "low_query_set":lowCount,
                "medium_query_set":mediumCount,
                "high_query_set":highCount,
                "critical_query_set":criticalCount,
                "applicant_list":applicant_list,        
    } 
    return render(request,"ml_fraudulent_app/index.html",context)




def pageview(request):
    pages = request.POST.get('pageno')
    #print(pages)

    #return render(request,'ml_fraudulent_app/index.html')
    return HttpResponse(simplejson.dumps({"success" : "true", "message" :"here" }, mimetype = "application/json"))


def ApplicantEdit(request):
    applicant_details = Applicant_Details.objects.all()
    context= {
        "applicant_details":applicant_details
    }
    return redirect(request,'ml_fraudulent_app/index.html',context)

def ApplicantUpdate(request,app_id):
    #dzero = Decimal(0)
    try:
        if request.method =="POST":
            app_id = request.POST.get('app_id')
            app_start_time = request.POST.get('app_start_time')
            app_submission_time = request.POST.get('app_submission_time')
            applicant_name = request.POST.get('applicant_name')
            app_email =  request.POST.get('app_email')
            app_onphone =  request.POST.get('app_onphone')
            app_ssn =  request.POST.get('app_ssn')
            app_mailing =  request.POST.get('app_mailing')
            renter =  request.POST.get('renter')
            unit_type =  request.POST.get('unit_type')
            requested_amount =  request.POST.get('requested_amount')
            geoLocation = request.POST.get('geoLocation')
            origin_ip = request.POST.get('origin_ip')
            classification =  request.POST.get('classification')

            reader = geoip2.database.Reader('ml_fraudulent_app/mmdb/GeoLite2-City.mmdb')
            final_ip= origin_ip.replace('\r', '')
            try:
                response = reader.city(final_ip)
                country_name= (response.country.name)
                print('country.name:{}'.format(country_name))
            except:
                pass
           

            #classification = Applicant_Details.objects.values('risk_table__classification')

            applicant_details = Applicant_Details(
                app_id = app_id,
                app_start_time = app_start_time,
                app_submission_time = app_submission_time,
                applicant_name = applicant_name,
                app_email = app_email,
                app_onphone = app_onphone,
                app_ssn =app_ssn,
                app_mailing = app_mailing,
                renter = renter,
                unit_type = unit_type,
                requested_amount =requested_amount,
                origin_ip= origin_ip,
                geoLocation=country_name,
                classification =classification,
            )
            

            risk = Risk_Table(
                app_id = app_id,
                classification =classification,
            )
            
            
           
            



    except Applicant_Details.DoesNotExist:
        raise Http404('Applicant Details does not exist')
    
    applicant_details.save()
    risk.save(update_fields=['classification'])
    messages.add_message(request,messages.INFO,'Data has been updated Successfully !')

    return redirect('home')
    #return redirect("ml_fraudulent_app/index.html")

def logout(request):
    return render(request,"ml_fraudulent_app/logout.html")


def reset_applicant(request):
    if request.method == "GET":
        dest = Applicant_Details.objects.all()
        #dest.delete()
        input_table = Input_Table.objects.all()
        print(input_table)
        
        return render(request, "ml_fraudulent_app/index.html")

def output(request):
    try:
        connection = mysql.connector.connect(host='127.0.0.1',
                                         database='ml_fraud_detection',
                                         user='root',
                                         password='Admin')
        cursor = connection.cursor()
        print("Applicant Details Table Data before deletion")
        sql_Delete_query = "DELETE FROM ml_fraudulent_app_applicant_details"
        cursor.execute(sql_Delete_query )
        connection.commit()
        print('number of rows deleted: ', cursor.rowcount)

        # for insert 
        mySql_insert_query = """INSERT INTO ml_fraudulent_app_applicant_details (app_id, app_start_time,app_submission_time, applicant_name, app_email, app_onphone, app_ssn, app_mailing, renter, unit_type, requested_amount, origin_ip, classification, geoLocation)

        SELECT app_id, app_start_time, app_submission_time, applicant_name, app_email, app_onphone, app_ssn, app_mailing, renter, unit_type, requested_amount, origin_ip, classification, geoLocation FROM ml_fraudulent_app_input_table"""

        cursor = connection.cursor()
        cursor.execute(mySql_insert_query)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Applicant Details table from Input Table")
        cursor.close()
        #messages.success(request, 'Table has been reseted...')

    except mysql.connector.Error as error:
        print("Failed to delete record from table: {}".format(error))
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


    db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Admin",
    database="ml_fraud_detection"
    )
    my_database = db_connection.cursor()

    df = pd.read_sql("select * from ml_fraud_detection.ml_fraudulent_app_applicant_details", db_connection)
    pd.set_option('display.expand_frame_repr', False)
    db_connection.close()

    df['app_start_time'] = df['app_start_time'].astype(str)
    df['app_submission_time'] = df['app_submission_time'].astype(str)
    df['app_id'] = df['app_id'].astype(int)
    df = df.replace("", np.nan, regex=True)
    df.columns = ['Application ID', 'Application Start Time', 'Application End Time', 'Applicant Name', 'Applicant Email', 'Applicant Phone     #', 'Applicant Social Security #',
                  'Applicant Address', 'Renter Quality', 'Unit Type', 'Requested Amount', 'Originating IP Address', 'Classification',   "Originating Country", 'Edited']

    df['Application Start Time'] = pd.to_datetime(df['Application Start Time'])
    df['Application End Time'] = pd.to_datetime(df['Application End Time'])
    df['Application Duration'] = (
        df['Application End Time'] - df['Application Start Time']) / np.timedelta64(1, 's')
    app_count = df.groupby('Applicant Social Security #').size().reset_index()
    app_count.columns = ['Applicant Social Security #',
                         'Application Count From Applicant']
    # merge back to the orginal df table
    df = df.merge(app_count, on='Applicant Social Security #', how='left')
    # setting up training dataset
    df_training = df.dropna(subset=['Classification'])
    feature_cols = ['Application Duration', 'Requested Amount',
                    'Originating Country', 'Application Count From Applicant']
    X_training = df_training[feature_cols]  # Features
    Y_training = pd.DataFrame(df_training['Classification'])  # Target variable
    X_all = df[feature_cols]
    # transfer categorical data to dummy variable for both dataset
    x_training_dummy = pd.get_dummies(X_training, drop_first=True)
    x_all_dummy = pd.get_dummies(X_all, drop_first=True)
    # prevent missing columns from all dataset and x traiing
    missing_cols = set(x_all_dummy.columns) - set(x_training_dummy.columns)
    for c in missing_cols:
        x_training_dummy[c] = 0

    x_training_dummy = x_training_dummy[x_training_dummy.columns]
    # re-name the columns to be more readable later
    x_training_dummy.columns = x_training_dummy.columns.str.replace('_', ' ')
    x_all_dummy.columns = x_all_dummy.columns.str.replace('_', ' ')
    # the following code will be used if we want to truning the max depth and min sample leaf in the future
    # cross validation from the max depth and min sample leaf
    #parameters = {'max_depth':range(1,20),'min_samples_leaf':range(1,20)}
    #clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
    #clf.fit(X=x_training_dummy, y=Y_training)
    #tree_model = clf.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(
        x_training_dummy, Y_training, test_size=1, random_state=1)  # 100% training can be adjust later
    clf = DecisionTreeClassifier(max_depth=len(feature_cols), min_samples_leaf=1)
    tree_model = clf.fit(X_train, y_train)
    # tree_model.score(X_train,y_train)
    # check the score performance as we are using 100% training data for now.
    # Predict the response for the whole dataset
    y_pred_class = tree_model.predict(x_all_dummy)
    y_prob_prob = pd.DataFrame(tree_model.predict_proba(x_all_dummy)).iloc[:, 0]
    # getting the decision critiria
    n_nodes = tree_model.tree_.node_count
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right
    feature = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold


    def find_path(node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (children_left[node_numb] != -1):
            left = find_path(children_left[node_numb], path, x)
        if (children_right[node_numb] != -1):
            right = find_path(children_right[node_numb], path, x)
        if left or right:
            return True
        path.remove(node_numb)
        return False


    def get_rule(path, column_names):
        mask = ''
        for index, node in enumerate(path):
            # We check if we are not in the leaf
            if index != len(path)-1:
                # Do we go under or over the threshold ?
                if (children_left[node] == path[index+1]):
                    mask += "{} <= {}\t".format(
                        column_names[feature[node]], round(threshold[node], 2))
                else:
                    mask += "{} > {}\t".format(
                        column_names[feature[node]], round(threshold[node], 2))
        # We insert the & at the right places
        mask = mask.replace("\t", ", ", mask.count("\t") - 1)
        mask = mask.replace("\t", "")
        return mask


    # apply the leave id for each record on the dataset
    leave_id = tree_model.apply(x_all_dummy)

    paths = {}
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(0, path_leaf, leaf)
        paths[leaf] = np.unique(np.sort(path_leaf))

    rules = {}
    for key in paths:
        rules[key] = get_rule(paths[key], x_all_dummy.columns)


    # formatting the decision rule

    decision_rule = pd.DataFrame(rules.values())
    decision_rule.columns = ['Decision']  # naming the variable as Decision
    decision_rule = pd.DataFrame(decision_rule['Decision'].str.split(
        ',', expand=True))  # spliting rules by ,
    decision_rule = decision_rule.fillna(value='None')  # fill out na space to None
    decision_rule = decision_rule.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)  # removing trailing and leading spaces
    # replace all string has < = 0.5 sign with a Not in front of it
    # round requested amount to 2 decimal
    # round application duratino to 1 decimal
    for i in range(len(decision_rule)):
        for j in range(len(decision_rule.columns)):
            if(decision_rule[j][i].__contains__('<= 0.5')):
                decision_rule[j][i] = decision_rule[j][i].replace('<= 0.5', '')
                decision_rule[j][i] = 'Not ' + decision_rule[j][i]
            if(decision_rule[j][i].__contains__('> 0.5')):
                decision_rule[j][i] = decision_rule[j][i].replace('> 0.5', '')
            if(decision_rule[j][i].__contains__('Requested Amount')):
                temp_cat = re.findall('[0-9.]+', decision_rule[j][i])[0]
                temp_num = str(
                    round(float(re.findall('[0-9.]+', decision_rule[j][i])[0]), 2))
                decision_rule[j][i] = re.sub(
                    temp_cat, temp_num, decision_rule[j][i])
            if(decision_rule[j][i].__contains__('Application Duration')):
                temp_cat = re.findall('[0-9.]+', decision_rule[j][i])[0]
                temp_num = str(
                    round(float(re.findall('[0-9.]+', decision_rule[j][i])[0])))
                decision_rule[j][i] = re.sub(
                    temp_cat, temp_num, decision_rule[j][i])

    # get the key out of dictionary


    def getList(dict):
        list = []
        for key in dict.keys():
            list.append(key)

        return list


    # rename the leave_id as the key from rules
    # setting a decision rule data copy for importance rank table only
    decision_rule_data = pd.DataFrame()
    decision_rule_data = decision_rule.copy()
    decision_rule_data['leave_id'] = getList(rules)

    # leave id for each record
    leave_id_df = pd.DataFrame(leave_id)
    leave_id_df.columns = ['leave_id']

    # setting up the final output data
    result = pd.DataFrame()
    result['Application ID'] = df['Application ID']
    result['Classification'] = df['Classification']
    result['Predict Class'] = y_pred_class
    result['Risk Score'] = round(y_prob_prob*100, 1)
    result['leave_id'] = leave_id_df['leave_id']

    final_result = pd.merge(result, decision_rule_data,
                            how='left', on='leave_id').drop('leave_id', axis=1)

    fraud_result = final_result[final_result['Predict Class'] == 'Fraud']
    total = fraud_result.shape[0]
    criteria_count = pd.DataFrame(fraud_result.iloc[:, 4:].stack(
    ).reset_index(level=1, drop=True).rename('Decision Criteria'))
    affected = pd.DataFrame(criteria_count.groupby(
        'Decision Criteria').size()).reset_index()
    affected.columns = ('Decision Criteria', 'Count')
    affected['Importance'] = (affected['Count']/total*100).astype(int)
    importance_rank = affected.drop(columns=['Count']).sort_values(
        'Importance', ascending=False).reset_index(drop=True)
    importance_rank = importance_rank[importance_rank['Decision Criteria'] != 'None']

    decision_rule_data = decision_rule_data.applymap(lambda x: x.strip(
    ) if isinstance(x, str) else x)  # removing all trailing and leading space
    # input varialbe
    num_feature_cols = ['Application Duration',
                        'Application Count From Applicant', 'Requested Amount']
    cat_feature_cols = ['Originating County']

    decision_rule_data = decision_rule_data.drop('leave_id', axis=1)

    # getting the decision criteria function


    def getting_decision_criteria(decision_rule_data, num_feature_cols, cat_feature_cols):
        import numpy as np
        import pandas as pd
        import re
        # get unique decision criteria from the decision path
        list = []
        for i in range(len(decision_rule_data)):
            list.append(decision_rule_data.iloc[i, :].unique())
        # flatten the nested list
        flat_list = [item for sublist in list for item in sublist]
        unique_list = np.unique(np.array(flat_list))
        # getting existing numerical variable in decision path
        existing_num_list = []
        for i in range(len(num_feature_cols)):
            existing_num_list.append(
                any(num_feature_cols[i] in s for s in unique_list))

        existing_num_variable = [existing_list*num_feature_cols for existing_list,
                                 num_feature_cols in zip(existing_num_list, num_feature_cols)]
        existing_num_variable = [
            name for name in existing_num_variable if len(name) > 0]

        # getting existing categorical variable in decision path

        existing_cat_list = []
        for i in range(len(cat_feature_cols)):
            existing_cat_list.append(
                any(cat_feature_cols[i] in s for s in unique_list))

        existing_cat_variable = [existing_list*cat_feature_cols for existing_list,
                                 cat_feature_cols in zip(existing_cat_list, cat_feature_cols)]
        existing_cat_variable = [
            name for name in existing_cat_variable if len(name) > 0]

        # numerical variable operation

        # setting up final output as an array
        final_num = [''] * len(decision_rule_data)

        # loop for every existing numerical variable
        for x in range(len(existing_num_variable)):

            name = existing_num_variable[x]
            name_less = name + ' <= '  # setting less than
            name_more = name + ' > '  # setting greater than

            # getting decision rule per row
            row_rule = [None] * len(decision_rule_data)
            # setting mathmatical upper bond
            upper = positive_infinity = float('inf')
            # setting mathmatical lower bond
            lower = negative_infinity = float('-inf')

            for i in range(len(decision_rule_data)):
                upper_bond_name = ''  # empty string for upper bond name
                lower_bond_name = ''  # empty string for lower bond name
                for j in range(len(decision_rule_data.columns)):
                    upper_bond = []  # list to store the current max number
                    lower_bond = []  # list to store the current min number

                    if (decision_rule_data.iloc[i][j].__contains__(name_less)):
                        # replace varibale string parts to '' and convert it to numeric
                        upper_bond.append(
                            float(decision_rule_data.iloc[i][j].replace(name_less, '')))
                        if len(upper_bond) > 0:  # if this upper bond exist
                            # compare the mathmacial upper bond to the decision upper bond, find the min of the two
                            upper_bond_name = name_less + \
                                str(min(min(upper_bond), upper))
                        # if there is no upper bond, then there is no upper bond string in the row rule
                        if len(upper_bond) == 0:
                            upper_bond_name = ''
                    # similar rules apply to lower bond
                    elif(decision_rule_data.iloc[i][j].__contains__(name_more)):
                        lower_bond.append(
                            float(decision_rule_data.iloc[i][j].replace(name_more, '')))
                        if len(lower_bond) > 0:
                            # compare the mathmacial lower bond to the decision lower bond, find the max of the two
                            lower_bond_name = name_more + \
                                str(max(max(lower_bond), lower))
                        if len(lower_bond) == 0:
                            lower_bond_name = ''
                # rounding the requested amount to decimal 2 and application duration for decimal 0 with both upeer and lower limit
                if(lower_bond_name.__contains__('Requested Amount')):
                    temp_cat = re.findall('[0-9.]+', lower_bond_name)[0]
                    temp_num = str(
                        round(float(re.findall('[0-9.]+', lower_bond_name)[0]), 2))
                    lower_bond_name = re.sub(temp_cat, temp_num, lower_bond_name)
                if(lower_bond_name.__contains__('Application Duration')):
                    temp_cat = re.findall('[0-9.]+', lower_bond_name)[0]
                    temp_num = str(
                        round(float(re.findall('[0-9.]+', lower_bond_name)[0])))
                    lower_bond_name = re.sub(temp_cat, temp_num, lower_bond_name)
                if(upper_bond_name.__contains__('Requested Amount')):
                    temp_cat = re.findall('[0-9.]+', upper_bond_name)[0]
                    temp_num = str(
                        round(float(re.findall('[0-9.]+', upper_bond_name)[0]), 2))
                    upper_bond_name = re.sub(temp_cat, temp_num, upper_bond_name)
                if(upper_bond_name.__contains__('Application Duration')):
                    temp_cat = re.findall('[0-9.]+', upper_bond_name)[0]
                    temp_num = str(
                        round(float(re.findall('[0-9.]+', upper_bond_name)[0])))
                    upper_bond_name = re.sub(temp_cat, temp_num, upper_bond_name)
                # combine upper and lower bond for single variable for each row
                row_rule[i] = lower_bond_name + upper_bond_name
                # formatting the final output for each numerical variable if the variable appearance more than once, we use seperater ',', if   not, we use only it self
                if row_rule[i].count(name) > 1:
                    temp_num = row_rule[i].index(name, 1)
                    row_rule[i] = row_rule[i][:temp_num] + \
                        ', ' + row_rule[i][temp_num:]
                elif row_rule[i].count(name) == 1:
                    row_rule[i] = row_rule[i]

                # formatting the final output for each row, if the variable appearance more than once, we use seperater ',', if not, we use     only it self
                if len(final_num[i]) == 0:
                    final_num[i] = row_rule[i] + final_num[i]
                elif len(final_num[i]) > 0:
                    final_num[i] = row_rule[i] + ', ' + final_num[i]

        # formatting the final output, remove any possible leading or tailing ','

        for i in range(len(final_num)):
            final_num[i] = re.sub(r"^[\, ]*", "", final_num[i])
            final_num[i] = re.sub(r"[\, ]*$", "", final_num[i])

        # categorical variable operation

        # setting up final categorical rule list
        final_cat = [''] * len(decision_rule_data)
        # row based categorical rule list
        row_rule_cat = [''] * len(decision_rule_data)

        for i in range(len(existing_cat_variable)):
            # for loop for every categorical variable
            cat_name = existing_cat_variable[i]
            categorical_df = decision_rule_data.mul(decision_rule_data.apply(lambda col: col.str.contains(
                cat_name, na=False), axis=1)).replace('', 'None')  # filter decision rule data frame with one categorical variable only
            # appearrance of the categorical variable
            count_cat_name = [''] * len(categorical_df.columns)
            # the appearance of the 'not word
            count_not_name = [''] * len(categorical_df.columns)

            for i in range(len(categorical_df)):
                for j in range(len(categorical_df.columns)):
                    # boolean list, true if detect the variable name, false else
                    count_cat_name[j] = categorical_df.iloc[i][j].__contains__(
                        cat_name)
                    count_not_name[j] = categorical_df.iloc[i][j].__contains__(
                        'Not')  # boolean list, true if detect 'Not', false else

                # if all appearing decision criteria has a not then # of true from count_cat_name will equal to # of true form the  count_not_name
                if sum(count_cat_name) == sum(count_not_name):
                    # selecting all true position variable. All Not criterials will be selcted for that varialbe
                    row_rule_temp = categorical_df.iloc[i].mul(count_cat_name)

                # if there is a positive critieria, # of true from the count_cat_name will greater than the # of true from the count_not name
                if sum(count_cat_name) > sum(count_not_name):
                    # reverse the count not name True to False, False to True
                    not_boolean_reverse = [not elem for elem in count_not_name]
                    # getting only the decision criteria that are positive for this variable only
                    row_rule_temp = categorical_df.iloc[i].mul(
                        not_boolean_reverse).replace('None', '')
                # remove all empty array from the singe row rule
                row_rule_cat[i] = [x for x in row_rule_temp if x]
            # format the final output
            # if there is no row rule then the final cat row will be empty
            for i in range(len(row_rule_cat)):
                if len(row_rule_cat[i]) == 0:
                    final_cat[i] = final_cat[i]
                # if there is row rule then the final cat will be append each row rule and seperate by ','
                if len(row_rule_cat[i]) > 0:
                    temp = len(row_rule_cat[i])
                    for j in range(temp):
                        final_cat[i] = row_rule_cat[i][j] + ', ' + final_cat[i]

        # formatting final output, removing any possible trailing and leading ','
        for i in range(len(final_cat)):
            final_cat[i] = re.sub(r"[\, ]*$", "", final_cat[i])
            final_cat[i] = re.sub(r"^[\, ]*", "", final_cat[i])

        # concatenate final numerical rule and final categorical rules together
        # setting up updated output
        updated_output = [''] * len(decision_rule_data)
        for i in range(len(decision_rule_data)):
            # concatenate numerical decision rules with categorical decision rule
            updated_output[i] = final_num[i] + ', ' + final_cat[i]
            # removing any possible trailing and leading ','
            updated_output[i] = re.sub(r"[\, ]*$", "", updated_output[i])
            updated_output[i] = re.sub(r"^[\, ]*", "", updated_output[i])

        # creating updated decision rule
        updated_decision_rule = pd.DataFrame(updated_output)

        def getList(dict):
            list = []
            for key in dict.keys():
                list.append(key)

            return list

        updated_decision_rule['leave_id'] = getList(rules)
        updated_decision_rule.columns = ['Decision Criteria', 'leave_id']
        return updated_decision_rule


    final_result = getting_decision_criteria(
        decision_rule_data, num_feature_cols, cat_feature_cols)
    updated_decision_table = pd.merge(
        result, final_result, how='left', on='leave_id').drop('leave_id', axis=1)

    updated_decision_table.rename(columns = {'Application ID' : 'app_id' , 'Classification' : 'classification', 'Predict Class' : 'Predict_Class' , 'Risk Score' : 'Risk_Score', 'Decision Criteria' : 'Decision_Criteria'}, inplace=True)
    print("================Table Records===============")
    print(updated_decision_table)  # ml_fraudulent_app_risk_table
    #Connection to DB
    con = MySQLdb.connect('localhost','root','Admin', 'ml_fraud_detection')
    my_sqlconn= create_engine("mysql+mysqldb://root:Admin@localhost/ml_fraud_detection")
    df = pd.DataFrame(data=updated_decision_table)
    with my_sqlconn.connect() as con:
        con.execute("TRUNCATE TABLE %s" % 'ml_fraudulent_app_risk_table')

    df.to_sql(name='ml_fraudulent_app_risk_table', con=my_sqlconn, if_exists='append',index=False)

    #df.to_sql(con=my_sqlconn,name='ml_fraudulent_app_risk_table',if_exists='replace',index=False)
    print("Risk table has been updated successfully !")
    importance_rank.rename(columns = {'Decision Criteria' : 'Decision_Criteria', 'Importance' : 'Importance'}, inplace=True)
    # connect to DB to store importance_rank values
    con = MySQLdb.connect('localhost','root','Admin', 'ml_fraud_detection')
    my_sqlconn= create_engine("mysql+mysqldb://root:Admin@localhost/ml_fraud_detection")
    df = pd.DataFrame(data=importance_rank)

    with my_sqlconn.connect() as con:
        con.execute("TRUNCATE TABLE %s" % 'ml_fraudulent_app_importance_rank_table')

    df.to_sql(name='ml_fraudulent_app_importance_rank_table', con=my_sqlconn, if_exists='append',index=False)


    #df.to_sql(con=my_sqlconn,name='ml_fraudulent_app_importance_rank_table',if_exists='replace',index=False)
    print("Importance Rank Table has been updated successfully !")

    print("================Imp Records===============")
    print(importance_rank)  # ml_fraudulent_app_importance_rank_table
    return HttpResponse("Reset Script Successfully Runned")
