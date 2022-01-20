from csv import reader
from urllib import response
import geoip2.database
from pprint import pprint
import mysql.connector
from mysql.connector import Error


try:
    connection = mysql.connector.connect(host='localhost',
                                         database='ml_fraud_detection',
                                         user='root',
                                         password='')
    if connection.is_connected():
        sql_select_Query = "SELECT * FROM ml_fraudulent_app_applicant_details"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        # get all records
        records = cursor.fetchall()
        print("Total number of rows in table: ", cursor.rowcount)
        print("\nOrigin IPs each row")
        reader = geoip2.database.Reader('mmdb/GeoLite2-City.mmdb')
        for row in records:
            ids= row[0]
            ips= row[11]
            final_ips= ips.replace('\r', '')
            try:
                print("Origin IP = ",final_ips)
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