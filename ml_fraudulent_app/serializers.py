from .models import Applicant_Details
from rest_framework import serializers

class Applicant_DetailsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Applicant_Details
        fields = ['app_id','app_start_time','app_submission_time','applicant_name','app_email','app_onphone','app_ssn','app_mailing','renter','unit_type','requested_amount','origin_ip','classification','AI_prediction','geoLocation']
       