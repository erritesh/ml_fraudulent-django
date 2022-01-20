from multiprocessing import context
from django import http
from django.http.response import HttpResponse
from django.shortcuts import redirect, render
from django.http import HttpResponse
from ml_fraudulent_app.models import Applicant_Details



# Create your views here.
def login(request):
    return render(request,"ml_fraudulent_app/login.html")
    
def home(request):
    applicant_details = Applicant_Details.objects.all()
    context = {"applicant_details":applicant_details}
    return render(request,"ml_fraudulent_app/index.html",context)

def ApplicantEdit(request):
    applicant_details = Applicant_Details.objects.all()
    context= {
        "applicant_details":applicant_details
    }
    return redirect(request,'ml_fraudulent_app/index.html',applicant_details)

def ApplicantUpdate(request,app_id):
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
            classification =classification,
            AI_prediction =AI_prediction
           
        )
        applicant_details.save()
        return redirect('home')
    return redirect("ml_fraudulent_app/index.html")

def applicant_details(request):
    return render(request,"ml_fraudulent_app/applicant_details.html")

def dashboard(request):
    applicant_details = Applicant_Details.objects.all()
    context = {"applicant_details":applicant_details}
    return render(request,"ml_fraudulent_app/dashboard.html",context)

def logout(request):
    return render(request,"ml_fraudulent_app/logout.html")

