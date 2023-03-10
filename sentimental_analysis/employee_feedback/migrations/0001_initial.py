# Generated by Django 4.1.5 on 2023-02-10 05:28

from django.db import migrations, models
import employee_feedback.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Employee_ID', models.CharField(max_length=10)),
                ('Domain', models.CharField(max_length=50)),
                ('Other_domain', models.CharField(blank=True, max_length=50, null=True)),
                ('Working_years', models.CharField(max_length=50)),
                ('Bored', models.CharField(max_length=10)),
                ('Free_time', models.CharField(blank=True, max_length=100, null=True)),
                ('Satisfied_with_company', models.CharField(max_length=10)),
                ('Improve_company', models.TextField(blank=True, null=True)),
                ('Recommend_friends', models.CharField(max_length=10)),
                ('Working_team', models.CharField(max_length=10)),
                ('Team_improve', models.TextField(blank=True, null=True)),
                ('Coming_to_work', models.TextField(blank=True, null=True)),
                ('Satisfied_with_manager', models.CharField(max_length=10)),
                ('Manager_improve', models.TextField(blank=True, null=True)),
                ('Culture_Values', models.CharField(max_length=10)),
                ('Compensation_Benefits', models.CharField(max_length=10)),
                ('Satisfied_with_management', models.CharField(max_length=10)),
                ('Management_improve', models.TextField(blank=True, null=True)),
                ('Improve_work', models.TextField(blank=True, null=True)),
                ('Satisfied_with_HR', models.CharField(max_length=10)),
                ('Hr_improve', models.TextField(blank=True, null=True)),
                ('Work_life_balance', models.CharField(max_length=10)),
                ('Suggestions', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Employee',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Employee_ID', models.CharField(max_length=10)),
                ('Employee_Firstname', models.CharField(max_length=100)),
                ('Employee_Lastname', models.CharField(max_length=100)),
                ('Employee_Domain', models.CharField(max_length=100)),
                ('Employee_Mail', models.CharField(max_length=50, validators=[employee_feedback.models.validate_mail])),
                ('Employee_Password', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Otp',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Employee_Mail', models.CharField(max_length=50, validators=[employee_feedback.models.validate_mail])),
                ('Employee_Otp', models.IntegerField()),
                ('Time', models.TimeField(null=True)),
            ],
        ),
    ]
