from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from myapp.views import index, upload_file,register, user_login, user_logout, predict_income, prediction_history

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', user_login, name='login'),
     path('predict_income/', predict_income, name='predict_income'),
    path('logout/', user_logout, name='logout'),
    path('', index, name='index'),
    path('upload/', upload_file, name='upload_file'),
    path('history/',prediction_history, name='prediction_history')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
