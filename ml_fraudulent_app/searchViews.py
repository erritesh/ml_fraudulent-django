from django.shortcuts import render
from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table

def searchView(request):
    applicant_data = Applicant_Details.objects.all().values('app_email', 'app_id', 'app_mailing', 'app_onphone', 'applicant_name', 'app_ssn', 'app_start_time', 'app_submission_time', 'applicant_name', 'classification', 'geoLocation', 'origin_ip', 'renter', 'requested_amount', 'risk_table__Decision_Criteria','risk_table__Risk_Score','risk_table__predict_class', 'unit_type')

    context = { 
        "applicant_data":applicant_data,
     }
    return render(request,"ml_fraudulent_app/searchView.html",context)