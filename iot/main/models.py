from django.db import models
from .validators import file_size


class Video(models.Model):
    # caption = models.CharField(max_length=100)
    video = models.FileField(upload_to="./", validators=[file_size]) # file_size : 업로드 파일 크기 제한
    # def __str__(self):
    #     return self.caption

# python manage.py makemigrations
# python manage.py migrate