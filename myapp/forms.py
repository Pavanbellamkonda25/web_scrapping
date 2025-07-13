from django import forms
from .models import *


class ScrapingForm(forms.Form):
    website_url = forms.URLField(label="Website URL")
    scraping_depth = forms.ChoiceField(
        choices=[("homepage", "Homepage Only"), ("all-pages", "All Pages")],
        label="Scraping Depth",
    )
    keywords = forms.CharField(required=False, label="Keywords to Monitor")


class SettingsForm(forms.Form):
    frequency = forms.ChoiceField(
        choices=[("daily", "Daily"), ("weekly", "Weekly"), ("monthly", "Monthly")],
        label="Scraping Frequency",
    )
    user_agent = forms.CharField(label="User Agent")
    rate_limit = forms.IntegerField(label="Rate Limit (requests per minute)")


# Scraped Data
class ScrapedDataForm(forms.ModelForm):
    class Meta:
        model = ScrapedData
        fields = ["url", "url_file", "title", "content"]
        widgets = {
            "url": forms.URLInput(
                attrs={"class": "form-control", "placeholder": "Enter URL"}
            ),
            "title": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Enter Title"}
            ),
            "content": forms.Textarea(
                attrs={"class": "form-control", "placeholder": "Enter Content"}
            ),
        }

# Text Analysis
class TextAnalysisForm(forms.ModelForm):
    class Meta:
        model = TextAnalysis
        fields = ["scraped_data", "sentiment", "keywords", "summary"]
        widgets = {
            "scraped_data": forms.Select(attrs={"class": "form-control"}),
            "sentiment": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Enter Sentiment"}
            ),
            "keywords": forms.Textarea(
                attrs={"class": "form-control", "placeholder": "Enter Keywords"}
            ),
            "summary": forms.Textarea(
                attrs={"class": "form-control", "placeholder": "Enter Summary"}
            ),
        }

# Task
class TaskForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = ["task_type", "status", "started_at", "completed_at", "result"]


# Search form
class SearchForm(forms.Form):
    query = forms.CharField(max_length=100)

    def clean_query(self):
        query = self.cleaned_data.get("query")
        # Custom validation logic
        return query
