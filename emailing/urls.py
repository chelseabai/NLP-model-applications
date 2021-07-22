from django.urls import path

from . import views

urlpatterns = [
    path("email", views.email, name="email"),
    path("excel", views.excel, name="excel"),
    path("advertise", views.advertise, name="advertise"),
    path("", views.gptmodel, name="model")]