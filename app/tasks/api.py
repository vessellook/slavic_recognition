from django.urls import re_path

from tasks import views

urlpatterns = [
    re_path('ping/?', views.PingView.as_view(), name='ping'),
    re_path('basic/binarize/?', views.BinarizeView.as_view(), name='basic-binarize'),
    re_path('basic/draw-lines/?', views.DrawLineBordersView.as_view(), name='basic-draw-lines'),
]

app_name = 'tasks'
