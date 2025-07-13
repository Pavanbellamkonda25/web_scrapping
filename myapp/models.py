from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.PositiveIntegerField(null=True, blank=True)  # Allow null values
    gender = models.CharField(
        max_length=10, choices=[("M", "Male"), ("F", "Female"), ("O", "Other")]
    )
    height = models.FloatField(null=True, blank=True)  # Allow null values
    weight = models.FloatField(null=True, blank=True)  # Allow null values

    def __str__(self):
        return self.user.username


# class ScrapedData(models.Model):
#     url = models.URLField(max_length=200)
#     url_file = models.FileField(default=''  
#            )
#     title = models.CharField(max_length=255)
#     content = models.TextField()
#     scraped_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return self.title




class ScrapedData(models.Model):
    task = models.ForeignKey("Task", on_delete=models.CASCADE, default = '')  
    url = models.URLField(max_length=200)
    url_file = models.FileField(blank=True, null=True)  
    title = models.CharField(max_length=255)
    content = models.TextField()
    scraped_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class TextAnalysis(models.Model):
    scraped_data = models.OneToOneField(
        ScrapedData, on_delete=models.CASCADE, related_name="analysis"
    )
    sentiment = models.CharField(max_length=50)
    keywords = models.TextField()
    summary = models.TextField()
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis of {self.scraped_data.title}"

class AnalysisResult(models.Model):
    task = models.ForeignKey("Task", on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=255)
    result_data = models.TextField()
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis {self.analysis_type} for Task {self.task.id}"


# class Task(models.Model):
#     TASK_TYPE_CHOICES = (
#         ("scraping", "Scraping"),
#         ("analysis", "Analysis"),
#     )
#     STATUS_CHOICES = (
#         ("pending", "Pending"),
#         ("in_progress", "In Progress"),
#         ("completed", "Completed"),
#         ("failed", "Failed"),
#     )

#     task_type = models.CharField(max_length=20, choices=TASK_TYPE_CHOICES)
#     status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
#     started_at = models.DateTimeField(blank=True, null=True)
#     completed_at = models.DateTimeField(blank=True, null=True)
#     result = models.TextField(blank=True, null=True)
#     progress = models.IntegerField(default=0)  # Add progress field

#     def __str__(self):
#         return f"{self.task_type} - {self.status}"
# from django.db import models


class Task(models.Model):
    task_type_choices = (
        ("scraping", "Scraping"),
        ("analysis", "Analysis"),
    )
    status_choices = (
        ("pending", "Pending"),
        ("in_progress", "In Progress"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    )

    task_type = models.CharField(max_length=20, choices=task_type_choices)
    status = models.CharField(max_length=20, choices=status_choices, default="pending")
    progress = models.IntegerField(default=0)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    result = models.TextField(blank=True, null=True)
    duration = models.DurationField(
        blank=True, null=True
    ) 
    pages_scraped = models.IntegerField(default=0)  # Number of pages scraped


class Log(models.Model):
    LOG_LEVEL_CHOICES = (
        ("info", "Info"),
        ("warning", "Warning"),
        ("error", "Error"),
    )

    timestamp = models.DateTimeField(auto_now_add=True)
    log_level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES)
    message = models.TextField()

    def __str__(self):
        return f"[{self.timestamp}] {self.log_level.upper()}: {self.message}"


class Configuration(models.Model):
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()

    def __str__(self):
        return f"{self.key}: {self.value}"

from django.db import models


class HarmfulContent(models.Model):
    website = models.URLField()
    website_url = models.URLField()
    type = models.CharField(max_length=255)
    detected_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.website} - {self.type}"

from django.db import models
from django.utils import timezone


class Report(models.Model):
    title = models.CharField(max_length=255)
    generated_at = models.DateTimeField(auto_now_add=True)
    file_url = models.URLField()  # URL to the generated report file
    filters = models.TextField(blank=True, null=True)  # Store filters as JSON or text

    def __str__(self):
        return f"{self.title} - {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}"


class ActivityLog(models.Model):
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.timestamp}: {self.description}"


class SystemHealth(models.Model):
    cpu_usage = models.FloatField()
    memory_usage = models.FloatField()
    storage_used = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"System Health at {self.timestamp}"
