from django.contrib.auth.models import User
from django.db import models

# Create your models here.
class Result(models.Model):
    result = models.CharField(max_length=255)

    def __str__(self):
        return self.result



