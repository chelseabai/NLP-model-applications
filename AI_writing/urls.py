from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("about", views.about, name="about"),
    path("writing", views.writing, name="writing"),
    path("chatting", views.chatting, name="chatting"),
    path("dungeon", views.dungeon, name="dungeon")]
