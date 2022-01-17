from django import http
from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from ml_fraudulent_app.models import Applicant_Details



# Create your views here.
def login(request):
    return render(request,"ml_fraudulent_app/login.html")
    
def home(request):
    return render(request,"ml_fraudulent_app/index.html")

#def applicant_details(request):
    #return render(request,"ml_fraudulent_app/applicant_details.html")

def dashboard(request):
    return render(request,"ml_fraudulent_app/dashboard.html")

def logout(request):
    return render(request,"ml_fraudulent_app/logout.html")

def show(request): 
    applicant_details = Applicant_Details.objects.filter()
    context = {"applicant_details":applicant_details}
    return render(request,"ml_fraudulent_app/show.html",context)