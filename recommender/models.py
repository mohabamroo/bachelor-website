# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db.models.signals import post_save
from django.db import models
import django.db.models.signals
from django.dispatch import receiver
from django.contrib.auth.models import User

# Create your models here.
class Profile(models.Model): #This extends the user class to add profile information
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    crawled = models.BooleanField(default=False, blank=True)

# a user model was just created! This now creates your extended user (a profile):
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        # instance is the user model being saved.
        Profile.objects.create(user=instance)


# a user model was just saved! This now saves your extended user (a profile):
@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
        instance.profile.save()