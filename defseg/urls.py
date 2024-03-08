from . import views
from django.urls import path 

urlpatterns = [
    path('', views.home, name='home'),
    # path('about_us', views.about_us, name='about_us'),
]
