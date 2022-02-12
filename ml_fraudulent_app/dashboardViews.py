from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table
from django.shortcuts import redirect, render
from django.db.models.functions import ExtractMonth
from django.db.models import Count
from django_pandas.io import read_frame
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go 
#from plotly.offline import iplot
import json
import plotly


def dashboard(request):
    applicant_details = Applicant_Details.objects.all().values('AI_prediction', 'app_email', 'app_id', 'app_mailing', 'app_onphone', 'app_ssn', 'app_start_time', 'app_submission_time', 'applicant_name', 'classification', 'geoLocation', 'origin_ip', 'renter', 'requested_amount', 'risk_table__Decision_Criteria', 'unit_type')

    total_query_set = Applicant_Details.objects.all().count()
    fraud_query_set = Applicant_Details.objects.filter(classification='Fraud').count()
    nonFraud_query_set = Applicant_Details.objects.filter(classification='Not_Fraud').count()
    pending_query_set = Applicant_Details.objects.filter(classification='').count()
    # Risk score count
    low_queryset = Applicant_Details.objects.filter(AI_prediction__range=(0.00,0.25)).count()
    medium_queryset = Applicant_Details.objects.filter(AI_prediction__range=(0.26,0.50)).count()
    high_queryset = Applicant_Details.objects.filter(AI_prediction__range=(0.51,0.75)).count()
    critical_queryset = Applicant_Details.objects.filter(AI_prediction__range=(0.76,1.00)).count()

    Pending_Predicted_Fraud = Risk_Table.objects.filter(classification='') & Risk_Table.objects.filter(predict_class='Fraud')
    pending_pre_fraud= Pending_Predicted_Fraud.count()
    Pending_Predicted_Not_Fraud = Risk_Table.objects.filter(classification='') & Risk_Table.objects.filter(predict_class='Not_Fraud')
    Pending_Pre_Not_Fraud = Pending_Predicted_Not_Fraud.count()

    # Risk Indicator
    reqested_amt =[]
    app_time = []
    amt_grt= []
    amt_less =[]
    time_grt=[]
    time_less =[]
     
    
    risk_table = Risk_Table.objects.all()
    for risk in risk_table:
        req_amt = str(risk.Decision_Criteria).split(",")

        if(len(req_amt)>1):
            reqested_amt.append(req_amt[0])
            app_time.append(req_amt[1])
 
        else:
            reqested_amt.append(req_amt[0])
            
    for y in reqested_amt:
        if (y.find('<=') != -1):
            amt_less.append(y)
        if (y.find('>')!=-1):
            amt_grt.append(y)
    finalAmtGrtVal= len(amt_grt)
    #print(finalAmtGrtVal)
    finalAmtLessVal= len(amt_less)
    
    for i in app_time:
        if (i.find('<=')!=-1):
            time_less.append(i)
        if(i.find('>')!=-1):
            time_grt.append(i)

    finalTimeGrtVal= len(time_grt)
    #print(finalTimeGrtVal)
    finalTimeLessVal= len(time_less)
    #print(finalTimeLessVal)
    
    sumoftotal = finalAmtGrtVal+finalAmtLessVal+finalTimeGrtVal+finalTimeLessVal
    
    per_amt_grt = round(((finalAmtGrtVal *100)/sumoftotal),2)
    per_amt_less = round(((finalAmtLessVal *100)/sumoftotal),2)
    per_time_grt = round(((finalTimeGrtVal *100)/sumoftotal),2)
    per_time_less = round(((finalTimeLessVal *100)/sumoftotal),2)

    # Risk Module Accuracy
    riskCount = risk_table.count()
    count = 0
    matchcount=0
    lstRiskAccu=[]
    lstRiskPre =[]

    riskAccu=  Risk_Table.objects.values('classification')
    riskPre= Risk_Table.objects.values('predict_class')
    riskaccuracy = (Risk_Table.objects.filter(classification='Fraud') | Risk_Table.objects.filter(classification='Not_Fraud')).count()
    
    for x in riskAccu:
        lstRiskAccu.append(x)
    for y in riskPre:
        lstRiskPre.append(y)
    for x in lstRiskAccu:
        if x.get('classification').strip()== lstRiskPre[count].get('predict_class'):
            matchcount+=1
        count+=1
    permatchcount= round(((matchcount*100)/count),2)
    
    # End risk Module Accuracy

    # month wise applicant data 
    monthlyapp = Applicant_Details.objects.annotate(month=ExtractMonth('app_submission_time')).values('month').annotate(c=Count('app_id')).order_by('month')
    # For map
    totalCount = Applicant_Details.objects.all().count()
    df = read_frame(applicant_details)
    df_count= df.groupby('geoLocation').size().reset_index()
    df_count.columns = ('Country','Sub_Count')

    df['percent'] = ((df_count['Sub_Count']/totalCount) * 100)
       
    data = dict(
        type = 'choropleth',
        colorscale = 'jet',
        locations = df_count['Country'],
        locationmode = "country names",
        z = df['percent'],
        text = df_count['Country'],
        colorbar = {'title' : "% of Application"},   
      )

    layout = dict(
              geo = dict(projection = {'type':'natural earth'}),
                width=600, 
                height=700,
                autosize=True,
                #paper_bgcolor="LightSteelBlue", 
             )
    
    choromap = go.Figure(data = [data],layout = layout)
    choromap.data[0].update(zmin=0, zmax=100,colorbar_len=0.80)
    choromap.update_layout(width=950,height=600, margin={"r":10,"t":0,"l":20,"b":10})
    
    jsondata = json.dumps(choromap, cls=plotly.utils.PlotlyJSONEncoder)

    context = {"applicant_details":applicant_details,
                "total_query_set":total_query_set,
                "fraud_query_set":fraud_query_set,
                "nonFraud_query_set":nonFraud_query_set,
                "pending_query_set":pending_query_set,
                "low_queryset":low_queryset,
                "medium_queryset":medium_queryset,
                "high_queryset":high_queryset,
                "critical_queryset":critical_queryset,
                "monthlyapp":monthlyapp,
                "jsondata":jsondata,
                "finalAmtGrtVal":finalAmtGrtVal,
                "finalAmtLessVal":finalAmtLessVal,
                "finalTimeGrtVal":finalTimeGrtVal,
                "finalTimeLessVal":finalTimeLessVal,
                "per_amt_grt":per_amt_grt,
                "per_amt_less":per_amt_less,
                "per_time_grt":per_time_grt,
                "per_time_less":per_time_less,
                "riskaccuracy":riskaccuracy,
                "permatchcount": permatchcount,
                "pending_pre_fraud":pending_pre_fraud,
                "Pending_Pre_Not_Fraud":Pending_Pre_Not_Fraud,         
        }
    print(per_amt_less)
    return render(request,"ml_fraudulent_app/dashboard.html",context)