from os import name
from django.contrib import admin
from django.urls import path, include,re_path
from ml_fraudulent_app import searchViews, views,dashboardViews


urlpatterns = [
    path('login',views.login,name="login"),
    path('', views.home, name="home"),
    path('search', searchViews.searchView, name="searchView"),
    path('dashboard', dashboardViews.dashboard, name="dashboardView"),
    path('logout/', views.logout, name="logout/"),
    #path('show',views.show),
    path('applicantEdit',views.ApplicantEdit,name='applicantEdit'),
    path('applicantUpdate/<str:app_id>',views.ApplicantUpdate,name='applicantUpdate'),
    #path('handler404',views.handler404,name="handler404"),
    path('reset_applicant', views.reset_applicant,name='reset_applicant'),
    path('output',views.output,name="output"),
    path('geolocationView',views.geolocationView,name="geolocationView"),
    #path('pageview',views.pageview,name="pageview"),
    re_path(r'^pageview/$', views.pageview),    
]
