from django.db import models

class EmailCheck(models.Model):
    content = models.TextField()
    result = models.CharField(max_length=10000)
    checked_at = models.DateTimeField(auto_now_add=True)
