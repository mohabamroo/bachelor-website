from django.conf.urls import url
from . import views

app_name = 'users'

urlpatterns = [
    url(r'^register/$', views.UserRegister.as_view(), name="user-register"),
    url(r'^profile/$', views.UserProfile.as_view(), name="user-profile"),
]