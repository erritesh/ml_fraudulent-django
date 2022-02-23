# Generated by Django 4.0.1 on 2022-02-21 06:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Applicant_Details',
            fields=[
                ('app_id', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('app_start_time', models.DateTimeField(auto_now=True)),
                ('app_submission_time', models.DateTimeField(auto_now_add=True)),
                ('applicant_name', models.CharField(max_length=30)),
                ('app_email', models.CharField(max_length=40)),
                ('app_onphone', models.CharField(max_length=15)),
                ('app_ssn', models.CharField(max_length=15)),
                ('app_mailing', models.TextField()),
                ('renter', models.IntegerField(blank=True, null=True)),
                ('unit_type', models.CharField(max_length=100)),
                ('requested_amount', models.IntegerField(blank=True, null=True)),
                ('origin_ip', models.CharField(max_length=18)),
                ('classification', models.CharField(max_length=20)),
                ('geoLocation', models.CharField(blank=True, default='Null', max_length=30, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Input_Table',
            fields=[
                ('app_id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('app_start_time', models.DateTimeField(auto_now=True)),
                ('app_submission_time', models.DateTimeField(auto_now_add=True)),
                ('applicant_name', models.CharField(max_length=30)),
                ('app_email', models.EmailField(max_length=40)),
                ('app_onphone', models.CharField(max_length=15)),
                ('app_ssn', models.CharField(max_length=15)),
                ('app_mailing', models.TextField()),
                ('renter', models.IntegerField(blank=True, null=True)),
                ('unit_type', models.CharField(max_length=100)),
                ('requested_amount', models.DecimalField(decimal_places=2, max_digits=10)),
                ('origin_ip', models.CharField(max_length=20)),
                ('classification', models.CharField(max_length=30)),
                ('geoLocation', models.CharField(blank=True, default='Null', max_length=30, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Risk_Table',
            fields=[
                ('app', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='ml_fraudulent_app.applicant_details', unique=True)),
                ('classification', models.CharField(max_length=50)),
                ('predict_class', models.CharField(max_length=50)),
                ('Risk_Score', models.IntegerField(blank=True, null=True)),
                ('Decision_Criteria', models.CharField(max_length=255)),
            ],
            options={
                'managed': False,
            },
        ),
    ]
