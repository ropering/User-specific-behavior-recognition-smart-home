from django.contrib import admin
from .models import Video
# Register your models here.
# 관리자 페이지에서 사용할 수 있도록 등록하는 부분인듯 하다
admin.site.register(Video)