from os import name
from django.contrib import admin
from django.urls import path, include
from .import views


urlpatterns = [
    path('login',views.login,name="login"),
    path('', views.home, name="home"),
    path('applicant_details', views.applicant_details, name="applicant_details"),
    path('dashboard', views.dashboard, name="dashboard"),
    path('logout/', views.logout, name="logout/"),
    #path('show',views.show),
    path('applicantEdit',views.ApplicantEdit,name='applicantEdit'),
    path('applicantUpdate/<str:app_id>',views.ApplicantUpdate,name='applicantUpdate'),
    path('riskVal',views.riskVal,name="riskVal"),
    
    
]