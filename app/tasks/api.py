from django.urls import re_path

from tasks import views

urlpatterns = [
    re_path('ping/?', views.PingView.as_view(), name='ping'),
    re_path('basic/binarize/?', views.BinarizeView.as_view(), name='basic-binarize'),
]

app_name = 'tasks'
