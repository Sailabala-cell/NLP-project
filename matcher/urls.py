from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('match/', views.match_cv_jd, name='match_cv_jd'),
]