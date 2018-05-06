# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import json, os

from django.contrib.auth.models import User, Group
from django.shortcuts import render
from rest_framework import permissions, generics
from recommender.models import Profile
from recommender.serializers import  ProfileSerializer
from django.views.generic import TemplateView
from rest_framework import views, serializers
from rest_framework.response import Response
from rest_framework import status
from django.core.exceptions import ObjectDoesNotExist
from rest_framework.decorators import api_view, permission_classes
from pymongo import MongoClient
from json import JSONEncoder
from bson.objectid import ObjectId
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import smtplib
class MongoEncoder(JSONEncoder):
    def default(self, obj, **kwargs):
        if isinstance(obj, ObjectId):
            return str(obj)
        return JSONEncoder.default(obj, **kwargs)


def connect_to_mongo():
    uri = 'mongodb://mohabamroo:ghostrider1@ds241699.mlab.com:41699/bachelor'
    client = MongoClient(uri,
                         connectTimeoutMS=30000,
                         socketTimeoutMS=None,
                         socketKeepAlive=True)
    db = client.get_database()
    return db


@api_view(['GET'])
def recommendations(request):
    instance = Profile.objects.get(user=request.user)
    profile = ProfileSerializer(instance, context={'request': request}).data
    db = connect_to_mongo()
    recommendations = db['user_recommendations'].find_one({'user': request.user.username})
    recommendations = MongoEncoder().encode(recommendations)
    recommendations = json.loads(recommendations)
    user_problems = []
    for problem_id in recommendations['recommendations']:
        problem = db['problems'].find_one({'id': problem_id})
        problem = MongoEncoder().encode(problem)
        problem = json.loads(problem)
        user_problems.append(problem)
    return Response({"message": "Got some data!", "data": request.data, 'profile': profile, 'rec': recommendations, 'problems': user_problems})


def index(request):
    return render(request, 'index.html', context=None)

def sendNotificationEmail(request):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("mohab@deemalab.com", "mohab.abdelmeguid")
    msg_text = "Thank you for registering, click this link to get your recommendations.\n<a href='" + request.data['domain']+ "#/recommendations/'>Click here</a>"
    msg = MIMEMultipart('alternative')
    msg['To'] = request.data['email']
    msg['Subject'] = 'Codeforces Recommendations'
    mt_html = MIMEText(msg_text, 'html')
    msg.attach(mt_html)
    server.sendmail("mohab@deemalab.com", request.data['email'], msg.as_string())
    server.quit()

@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def process_user(request):
    limit = "900"
    print "running command in terminal"
    # os.system("python -c \"import recommender.cfuu; recommender.cfuu.process_new_user('" + request.data['username']+
    # "', '" + request.data['domain'] + "', " + limit + ");\"")
    import recommender.cfuu
    recommender.cfuu.process_new_user(request.data['username'], request.data['domain'], 900);
    print "finished terminal"
    sendNotificationEmail(request)
    return Response({"message": "Got some eeeeshta!"})

