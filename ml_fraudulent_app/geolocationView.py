from csv import reader
from django.http.response import HttpResponse
from django.shortcuts import redirect, render
from django.http import HttpResponse
import geoip2.database
from pprint import pprint
import mysql.connector
from mysql.connector import Error
from django.contrib import messages
from numpy import record
from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table
from django.shortcuts import get_list_or_404, get_object_or_404

def singlegeoLocationupdateView(request,app_id):
    cat = get_object_or_404(Applicant_Details, pk=app_id)
    print("Updating Geo Location")
    try:
        connection = mysql.connector.connect(host='localhost',
                                         database='ml_fraud_detection',
                                         user='root',
                                         password='Admin')
        if connection.is_connected():
            reader = geoip2.database.Reader('ml_fraudulent_app/mmdb/GeoLite2-City.mmdb')
            app_id = cat.app_id
            IPs= cat.origin_ip
            final_ip= IPs.replace('\r', '')
            try:
                    #print("Origin IP = ",final_ips)
                    response = reader.city(final_ip)
                    country_name= (response.country.name)
                    print('country.name:{}'.format(country_name))
                    # insert  record now
                    varlist = [country_name,app_id]
                    mysql_update_query = """UPDATE ml_fraudulent_app_applicant_details SET geoLocation = %s where app_id = %s"""
            
                    cursor = connection.cursor()
                    cursor.execute(mysql_update_query,varlist)
                    connection.commit()
                    print(cursor.rowcount, "Record inserted successfully into ml_fraudulent_app_applicant_details table")
                    cursor.close()
                    messages.add_message(request,messages.INFO,'Geo Location  has been updated Successfully !')

                    
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
    return redirect('home')
    
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
    #messages.add_message(request,messages.INFO,'Geolocation has been updated successfully!')
    #return HttpResponse (messages.add_message(request,messages.INFO,'Geolocation has been updated successfully!'))
    return HttpResponse("Geo Localtion Script Runned successfully ")       
