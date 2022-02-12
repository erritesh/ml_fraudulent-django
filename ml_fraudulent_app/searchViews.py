from django.shortcuts import render
from ml_fraudulent_app.models import Applicant_Details, Risk_Table, Input_Table

def searchView(request):
    applicant_data = Applicant_Details.objects.all()
    context = { 
        "applicant_data":applicant_data,
     }
    return render(request,"ml_fraudulent_app/searchView.html",context)