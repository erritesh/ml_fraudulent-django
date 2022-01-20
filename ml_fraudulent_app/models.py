from django.db import connections
from django.db import models
from datetime import datetime
from django.utils import timezone


# Create your models here.
class Applicant_Details(models.Model):
    app_id = models.CharField(max_length=100 ,
                        primary_key=True)
    app_start_time = models.DateTimeField(auto_now=True)
    app_submission_time = models.DateTimeField(auto_now_add=True)
    applicant_name = models.CharField(max_length=30)
    app_email = models.CharField(max_length=40)
    app_onphone = models.CharField(max_length=15)
    app_ssn = models.CharField(max_length=15)
    app_mailing = models.TextField()
    renter = models.IntegerField(max_length=10)
    unit_type = models.CharField(max_length=100)
    requested_amount = models.DecimalField(
                         max_digits = 10,
                         decimal_places = 2)
    origin_ip = models.CharField(max_length=18)
    classification = models.CharField(max_length=20)
    AI_prediction = models.DecimalField(
                         max_digits = 4,
                         decimal_places = 2)
    geoLocation = models.CharField(max_length=30, default='null')


    







def __str__(self):
     return self.title