from django.conf.urls import url
from . import views

app_name = 'recommender'

urlpatterns = [
    url(r'^new_user/$', views.process_user, name="process_new_user"),
    url(r'^recommendations/$', views.recommendations, name="recommendations"),
    url(r'^', views.index, name="angular-index"),

]