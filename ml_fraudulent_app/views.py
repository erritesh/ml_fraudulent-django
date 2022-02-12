from gettext import find
from django.http.response import HttpResponse
from django.shortcuts import redirect, render
from django.http import HttpResponse
from numpy import int64
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

# Create your views here.
def login(request):
    return render(request,"ml_fraudulent_app/login.html")
    
def home(request):
    total_query_set = Applicant_Details.objects.all().count()
    low_queryset = Applicant_Details.objects.filter(classification='') & Applicant_Details.objects.filter(AI_prediction__range=(0.00,0.25))
    medium_queryset = Applicant_Details.objects.filter(classification='') & Applicant_Details.objects.filter(AI_prediction__range=(0.26,0.50))
    high_queryset = Applicant_Details.objects.filter(classification='') & Applicant_Details.objects.filter(AI_prediction__range=(0.51,0.75))
    critical_queryset = Applicant_Details.objects.filter(classification='') & Applicant_Details.objects.filter(AI_prediction__range=(0.77,1.00))
    lowCount=low_queryset.count()
    mediumCount = medium_queryset.count()
    highCount = high_queryset.count()
    criticalCount = critical_queryset.count()
      
    #applicant_details = Applicant_Details.objects.get_queryset().order_by('app_id')

    applicant_details = Applicant_Details.objects.all().values('AI_prediction', 'app_email', 'app_id', 'app_mailing', 'app_onphone', 'app_ssn', 'app_start_time', 'app_submission_time', 'applicant_name', 'classification', 'geoLocation', 'origin_ip', 'renter', 'requested_amount', 'unit_type','risk_table__Decision_Criteria','risk_table__predict_class').order_by('app_id')

    # df = read_frame(applicant_details)
    # json_records = df.reset_index().to_json(orient ='records')
    # data = []
    # data = json.loads(json_records)
    

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
    pages = request.GET.get["selectedpage"]
    print(pages)

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
        AI_prediction = request.POST.get('AI_prediction')
          
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
            geoLocation=geoLocation,
            classification =classification,
            AI_prediction =AI_prediction
           
        )
        applicant_details.save()
        messages.add_message(request,messages.INFO,'Data has been updated Successfully !')
        return redirect('home')
    return redirect("ml_fraudulent_app/index.html")

def logout(request):
    return render(request,"ml_fraudulent_app/logout.html")

def geolocationView(request):
    try:
        connection = mysql.connector.connect(host='localhost',
                                         database='ml_fraud_detection',
                                         user='root',
                                         password='Admin')
        if connection.is_connected():
            sql_select_Query = "SELECT * FROM ml_fraudulent_app_applicant_details"
            cursor = connection.cursor()
            cursor.execute(sql_select_Query)
            # get all records
            records = cursor.fetchall()
            print("Total number of rows in table: ", cursor.rowcount)
            print("\nOrigin IPs each row")
            reader = geoip2.database.Reader('ml_fraudulent_app/mmdb/GeoLite2-City.mmdb')
            for row in records:
                ids= row[0]
                ips= row[11]
                final_ips= ips.replace('\r', '')
                try:
                    #print("Origin IP = ",final_ips)
                    response = reader.city(final_ips)
                    country_name= (response.country.name)
                    print('country.name:{}'.format(country_name))
                    # insert  record now
                    varlist = [country_name,ids]
                    print(type(varlist))
                    #mySql_insert_query = """INSERT INTO geolocation_demo (geo_location) VALUES (%s) """ % (tuple(varlist))
                    mysql_update_query = """UPDATE ml_fraudulent_app_applicant_details SET geoLocation = %s where app_id = %s"""
            
                    cursor = connection.cursor()
                    cursor.execute(mysql_update_query,varlist)
                    connection.commit()
                    print(cursor.rowcount, "Record inserted successfully into ml_fraudulent_app_applicant_details table")
                    cursor.close()
                except :
                    pass
            
    except Error as e:
        print("Failed to insert record into ml_fraudulent_app_applicant_details table {}".format(e))
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    reader.close()
    return HttpResponse("Script Runned")

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
        print("Applicant Details Table Data before deleted")
        sql_Delete_query = "DELETE FROM ml_fraudulent_app_applicant_details"
        cursor.execute(sql_Delete_query )
        connection.commit()
        print('number of rows deleted: ', cursor.rowcount)

        # for insert 
        mySql_insert_query = """INSERT INTO ml_fraudulent_app_applicant_details (app_id, app_start_time,app_submission_time, applicant_name, app_email, app_onphone, app_ssn, app_mailing, renter, unit_type, requested_amount, origin_ip, classification, AI_prediction, geoLocation)

        SELECT app_id, app_start_time, app_submission_time, applicant_name, app_email, app_onphone, app_ssn, app_mailing, renter, unit_type, requested_amount, origin_ip, classification, AI_prediction, geoLocation FROM ml_fraudulent_app_input_table"""

        cursor = connection.cursor()
        cursor.execute(mySql_insert_query)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Applicant Details table from Input Table")
        cursor.close()

    except mysql.connector.Error as error:
        print("Failed to delete record from table: {}".format(error))
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return HttpResponse("Script Runned")