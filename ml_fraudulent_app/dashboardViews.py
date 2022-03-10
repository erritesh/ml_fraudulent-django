from pkgutil import ImpImporter
from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table,Importance_rank_table
from django.shortcuts import redirect, render
from django.db.models.functions import ExtractMonth,ExtractYear
from django.db.models import Count
from django_pandas.io import read_frame
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go 
#from plotly.offline import iplot
import json
import plotly
from django.db.models import Sum
from django.db.models import F



def dashboard(request):
    applicant_details = Applicant_Details.objects.all().values('app_email', 'app_id', 'app_mailing', 'app_onphone', 'app_ssn', 'app_start_time', 'app_submission_time', 'applicant_name', 'classification', 'geoLocation', 'origin_ip', 'renter', 'requested_amount', 'risk_table__Decision_Criteria', 'unit_type')

    total_query_set = Applicant_Details.objects.all().count()
    fraud_query_set = Applicant_Details.objects.filter(classification='Fraud').count()
    nonFraud_query_set = Applicant_Details.objects.filter(classification='Legitimate').count()
    pending_query_set = Applicant_Details.objects.filter(classification='').count()
    # Risk score count
    low_queryset = Risk_Table.objects.filter(Risk_Score__range=(0,25)).count()
    medium_queryset = Risk_Table.objects.filter(Risk_Score__range=(26,50)).count()
    high_queryset = Risk_Table.objects.filter(Risk_Score__range=(51,75)).count()
    critical_queryset = Risk_Table.objects.filter(Risk_Score__range=(76,100)).count()

    Pending_Predicted_Fraud = Risk_Table.objects.filter(classification=None) & Risk_Table.objects.filter(predict_class='Fraud')
    pending_pre_fraud= Pending_Predicted_Fraud.count()
    Pending_Predicted_Legitimate = Risk_Table.objects.filter(classification=None) & Risk_Table.objects.filter(predict_class='Legitimate')
    Pending_Pre_Not_Fraud = Pending_Predicted_Legitimate.count()

    # Risk Indicator
    reqested_amt =[]
    app_time = []
    amt_grt= []
    amt_less =[]
    time_grt=[]
    time_less =[]
    finalAmtGrtVal = 0
    finalAmtLessVal = 0
    finalTimeGrtVal = 0
    finalTimeLessVal = 0 
    
    risk_table = Risk_Table.objects.all()
   
    if not risk_table:
        print("No data in risk table")
        pass
        
    else:
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
    
    # Risk Module Accuracy
    riskCount = risk_table.count() # 0 
    count = 0
    matchcount=0
    lstRiskAccu=[]
    lstRiskPre =[]
    permatchcount =0

    #User.objects.filter(last_name=F("first_name"))

    riskAccu=  Risk_Table.objects.values('classification')
    riskPre= Risk_Table.objects.values('predict_class')

    if not riskAccu:
        print("No data in Classification Column in Risk Table")
        pass
    else :
        
        for x in riskAccu:
            lstRiskAccu.append(x)
        for y in riskPre:
            lstRiskPre.append(y)
        for x in lstRiskAccu:
            if x.get('classification') is None:

                pass
            else:
                if x.get('classification').strip()== lstRiskPre[count].get('predict_class'):
                  matchcount+=1
                count+=1 #8188
    
    classification_val=  Risk_Table.objects.filter(classification__isnull=False).count()
    finaltab= Risk_Table.objects.filter(classification=F('predict_class')).count()

    permatchcount= round(((finaltab*100)/classification_val),2)
        
    # End risk Module Accuracy

    # month wise applicant data 
    monthlyapp = Applicant_Details.objects.annotate(month=ExtractMonth('app_submission_time')).values('month').annotate(c=Count('app_id')).order_by('month')
    print(type(monthlyapp))

    yearlyapp = Applicant_Details.objects.annotate(year=ExtractYear('app_submission_time')).values('year').annotate(c=Count('app_id')).order_by('year')

    # For map
    totalCount = Applicant_Details.objects.all().count()
    df = read_frame(applicant_details)
    df_count= df.groupby('geoLocation').size().reset_index()
    df_count.columns = ('Country','Sub_Count')

    df['percent'] = round(((df_count['Sub_Count']/totalCount) * 100),2)
     
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
    choromap.update_layout(width=1050,height=600, margin={"r":0,"t":0,"l":20,"b":10})
    
    jsondata = json.dumps(choromap, cls=plotly.utils.PlotlyJSONEncoder)

    # From Imporance Rank table 
    importance_rank = Importance_rank_table.objects.all().values('ImportanceID','Decision_Criteria','Importance','Count')


    fraud_ReqAmount = Applicant_Details.objects.filter(risk_table__predict_class='Fraud').values('requested_amount').aggregate(Sum('requested_amount'))
    fraud_amount= fraud_ReqAmount.values()
    print(type(fraud_amount))
    #final_fraud_ReqAmount= int(float(fraud_ReqAmount))

    legit_ReqAmount = Applicant_Details.objects.filter(risk_table__predict_class='Legitimate').values('requested_amount').aggregate(Sum('requested_amount'))
    #finallegit_reqAmount= int(float(legit_ReqAmount))
    

    context = {
                "applicant_details":applicant_details,
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
                # Risk Indicators context value
                "finalAmtGrtVal":finalAmtGrtVal, # RI - 3 , 4
                "finalAmtLessVal":finalAmtLessVal, # RI - 1
                "finalTimeGrtVal":finalTimeGrtVal, # RI -2 
                "finalTimeLessVal":finalTimeLessVal, # RI - 5 , 6 
                # "per_amt_grt":per_amt_grt,
                # "per_amt_less":per_amt_less,
                # "per_time_grt":per_time_grt,
                # "per_time_less":per_time_less,
                #"riskaccuracy":riskaccuracy,
                "permatchcount": permatchcount,
                "pending_pre_fraud":pending_pre_fraud,
                "Pending_Pre_Not_Fraud":Pending_Pre_Not_Fraud,
                "importance_rank":importance_rank,
                "yearlyapp":yearlyapp,         
        }
    return render(request,"ml_fraudulent_app/dashboard.html",context)
