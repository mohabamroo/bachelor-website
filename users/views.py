# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from rest_framework import permissions, generics

from django.shortcuts import render
from recommender.models import Profile
from recommender.serializers import UserSerializer, ProfileSerializer, RegistrationSerializer, GroupSerializer
from django.contrib.auth.models import User, Group
from rest_framework import viewsets

# Create your views here.


class UserRegister(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegistrationSerializer
    permission_classes = [permissions.AllowAny]


class UserProfile(generics.RetrieveAPIView):
    queryset = Profile.objects.all()
    serializer_class = ProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        try:
            profile = Profile.objects.get(user=self.request.user)
            return profile
        except ObjectDoesNotExist:
            return Profile.objects.create(user=self.request.user)

    def get_queryset(self):
        return Profile.objects.filter(user=self.request.user)


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
