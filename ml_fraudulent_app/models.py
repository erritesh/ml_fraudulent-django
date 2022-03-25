from email.policy import default
from pyexpat import model
from django.db import connections
from django.db import models
from sqlalchemy import null
#from decimal import Decimal


# Create your models here.
class Applicant_Details(models.Model):
    SELECTCATEGORY = 0
    FRAUD = 1
    NOTFRAUD = 2
    CATEGORY_TYPES = (
        (SELECTCATEGORY, 'Select Category'),
        (FRAUD, 'Fraud'),
        (NOTFRAUD, 'Not-Fraud'),
    )
    app_id = models.CharField(max_length=20 ,
                        primary_key=True)
    app_start_time = models.DateTimeField(auto_now=True)
    app_submission_time = models.DateTimeField(auto_now_add=True)
    applicant_name = models.CharField(max_length=30)
    app_email = models.CharField(max_length=40)
    #app_email = models.models.EmailField(max_length=40)
    app_onphone = models.CharField(max_length=15)
    app_ssn = models.CharField(max_length=15)
    app_mailing = models.TextField()
    renter = models.IntegerField(blank=True, null=True)
    unit_type = models.CharField(max_length=100)
    # requested_amount = models.DecimalField(
    #                      max_digits = 10,
    #                      decimal_places = 2)
    requested_amount = models.DecimalField(max_digits = 10,
                          decimal_places = 2,default=0)
    origin_ip = models.CharField(max_length=18)
    classification = models.CharField(max_length=20)
    #classification = models.PositiveSmallIntegerField(choices=CATEGORY_TYPES)
    
    geoLocation = models.CharField(max_length=30, null=True, blank=True, default='Null')
    edited= models.BooleanField(default=0)
    #def __str__(self):
    #    return "%s %s" % (self.app_id, self.applicant_name)

class Risk_Table(models.Model):
    #app_id = models.AutoField(primary_key=True)
    app = models.ForeignKey(Applicant_Details, primary_key=True,unique=True, blank=True, on_delete=models.CASCADE)
    classification = models.CharField(max_length=50)
    predict_class = models.CharField(max_length=50)
    Risk_Score =models.DecimalField(
                          max_digits = 5,
                          decimal_places = 1,default=0)
    Decision_Criteria = models.CharField(max_length=1000)

    # class Meta:
    #     managed = False
        

    #def __str__(self):
    #   return "%s %s" % (self.app, self.predict_class)


class Input_Table(models.Model):
                    
    app_id = models.CharField(primary_key=True, max_length=10)
    app_start_time = models.DateTimeField(auto_now=True)
    app_submission_time = models.DateTimeField(auto_now_add=True)
    applicant_name = models.CharField(max_length=30)
    app_email = models.EmailField(max_length=40)
    app_onphone = models.CharField(max_length=15)
    app_ssn = models.CharField(max_length=15)
    app_mailing = models.TextField()
    renter = models.IntegerField(blank=True, null=True)
    unit_type = models.CharField(max_length=100)
    requested_amount = models.DecimalField(
                         max_digits = 10,
                         decimal_places = 2,default=0)
    origin_ip = models.CharField(max_length=20)
    classification = models.CharField(max_length=30)
    
    geoLocation = models.CharField(max_length=30, null=True, blank=True, default='Null')

    def __str__(self):
        return "%s %s" % (self.app_id, self.applicant_name)

class Importance_rank_table(models.Model):
    ImportanceID = models.AutoField(primary_key=True, 
                                    null=False,blank=False)
    Decision_Criteria = models.CharField(max_length=1000) 
    Count = models.IntegerField(default=0)                 
    Importance = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    
    def __str__(self):
        return "%s %s" % (self.Importance_id, self.Decision_Criteria)
    