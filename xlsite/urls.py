from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.list, name='list'),
    path('getFeatures', views.getFeatures, name='getFeatures'),
    path('seeFiles', views.seeFiles, name='seeFiles'),
    path('classify', views.findGenre, name='classify'),
    path('convert', views.convTo, name='convert'),
   ]