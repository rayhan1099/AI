from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/chat/', views.chat, name='chat'),
    path('api/history/<str:session_id>/', views.history, name='history'),
]

