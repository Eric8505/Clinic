from django.db import models

# Create your models here.


class Patient(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField()
    insurance_info = models.TextField()
    access_id = models.CharField(max_length=100)
    patient_image = models.ImageField(upload_to='patients/')

    def __str__(self):
        return f"{self.first_name} {self.last_name}"
