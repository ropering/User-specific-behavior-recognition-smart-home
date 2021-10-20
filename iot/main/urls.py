from django.urls import path
from . import views

urlpatterns = [
    path('request_face/<id>', views.request_face),
    path('get_recognition', views.get_recognition),

    path('train/', views.face_train),
    path('face_test/', views.face_recognition),
    path('finger_test/', views.finger_recognition),
    path('upload/<id>/', views.upload),
    path('index/<id>', views.index),
    path('recognition_behavior/', views.recognition_behavior),
    path('recognition_face/', views.recognition_face),
    # path('streaming_test/', views.streaming_test),
]
#
# RuntimeError: You called this URL via POST, but the URL doesn
# 't end in a slash and you have APPEND_SLASH set. Django can't
#  redirect to the slash URL while maintaining POST data. Chang
# e your form to point to 172.30.1.4:5000/upload/3/ (note the t
# railing slash), or set APPEND_SLASH=False in your Django sett
# ings.
