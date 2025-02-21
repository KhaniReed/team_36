from django.urls import path 
from . import views

urlpatterns = [
    path('', views.Home, name ='home'),
    path('developer/', views.developer, name='developer'),
    path('detection/', views.detection, name='detection'),
    path('references/', views.references, name='references'),
]