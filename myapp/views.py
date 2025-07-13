from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseNotFound
from .forms import *
from .models import *
import requests
from bs4 import BeautifulSoup
import json
import csv
from django. utils import timezone

# Create your views here.


# scraping# views.py
import csv
from django.shortcuts import render, redirect
from django.utils import timezone
from .forms import ScrapingForm
from .models import Task


from django.core.files.storage import FileSystemStorage

# scrapingimport csv
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.utils import timezone
from .models import Task


# scraping
def start_scraping_task(request):
    if request.method == "POST":
        # Handle form data
        url = request.POST.get("website_url")
        depth = request.POST.get("scraping_depth")
        keywords = request.POST.get(
            "keywords", ""
        )  # Default to an empty string if not provided
        url_file = request.FILES.get("url_file")

        urls = []

        # If a file is uploaded, process the CSV file for URLs
        if url_file:
            # Save the uploaded file
            fs = FileSystemStorage()
            filename = fs.save(url_file.name, url_file)
            file_path = fs.path(filename)

            # Read the CSV file
            with open(file_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:  # Ensure row isn't empty
                        urls.append(row[0])

        # If no file is uploaded, use the single URL input (if provided)
        if url:
            urls.append(url)

        # If no URLs are provided from either method, return an error
        if not urls:
            return render(
                request,
                "scraping.html",
                {
                    "form": ScrapingForm(),
                    "error": "Please provide at least one URL or upload a list.",
                },
            )

        # Create a new scraping task for the single URL
        task = Task.objects.create(
            task_type="scraping",
            status="in_progress",
            result=f"Started scraping {url}",
            started_at=timezone.now(),
        )
        print(f"Task created: {task}")

        # Start scraping in the background (this should ideally be done asynchronously)
        scrape_website(url, depth, keywords.split(","), task)

        return redirect("completed_scraping_tasks")
    else:
        form = ScrapingForm()

    return render(request, "scraping.html", {"form": form})


# def ongoing_scraping_tasks(request):
#     tasks = Task.objects.filter(task_type="scraping", status="in_progress")
#     print("Ongoing tasks:", tasks)
#     return render(request, "ongoing_tasks.html", {"tasks": tasks})
def download_data(request, task_id):
    # Fetch data related to the task
    task = get_object_or_404(Task, id=task_id)
    # results = ScrapedData.objects.filter(task=task)
    results = ScrapedData.objects.filter(url__icontains=task.result.split(" ")[-1])

    # Create an HTTP response with CSV content
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="scraped_data_{task_id}.csv"'
    )

    writer = csv.writer(response)
    writer.writerow(["URL", "Title", "Content"])  # Header row
    for result in results:
        writer.writerow([result.url, result.title, result.content])

    return response


# def download_data(request, task_id):
#     # Get the data related to the task
#     data = ScrapedData.objects.filter(task_id=task_id)

#     # Create a CSV response
#     response = HttpResponse(content_type="text/csv")
#     response["Content-Disposition"] = (
#         f'attachment; filename="scraped_data_{task_id}.csv"'
#     )

#     writer = csv.writer(response)
#     writer.writerow(["URL", "Title", "Content"])

#     for item in data:
#         writer.writerow([item.url, item.title, item.content])

#     return response


def ongoing_scraping_tasks(request):
    ongoing_tasks = Task.objects.filter(task_type="scraping", status="in_progress")
    completed_tasks = Task.objects.filter(task_type="scraping", status="completed")
    return render(
        request,
        "ongoing_tasks.html",
        {"ongoing_tasks": ongoing_tasks, "completed_tasks": completed_tasks},
    )


def completed_scraping_tasks(request):
    tasks = Task.objects.filter(task_type="scraping", status="completed")
    return render(request, "completed_tasks.html", {"tasks": tasks})

from django.shortcuts import render, get_object_or_404
# def view_results(request, task_id):
#     task = get_object_or_404(Task, id=task_id)
#     # Assuming you have a related model `ScrapedData` for storing results
#     results = ScrapedData.objects.filter(task=task)
#     return render(request, "view_results.html", {"task": task, "results": results})


def view_results(request, task_id):
    try:
        task = Task.objects.get(id=task_id)
        results = ScrapedData.objects.filter(task=task)  # Now this will work
        context = {
            "task": task,
            "results": results,
        }
        return render(request, "view_results.html", context)
    except Task.DoesNotExist:
        return redirect("start_scraping_task")  # Redirect if task doesn't exist

def view_results(request, task_id):
    try:
        task = Task.objects.get(id=task_id)
        results = ScrapedData.objects.filter(task=task)  # Query data associated with the task

        if not results.exists():
            return render(request, 'view_results.html', {
                'task': task,
                'results': None,
                'message': 'No data available for this task.'
            })

        return render(request, 'view_results.html', {
            'task': task,
            'results': results
        })

    except Task.DoesNotExist:
        return redirect('start_scraping_task')  # Handle the case when the task is not found


# def scrape_website(url, depth, keywords, task):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, "html.parser")

#         title = soup.title.string if soup.title else "No title"
#         content = soup.get_text()

#         # Save scraped data
#         ScrapedData.objects.create(url=url, title=title, content=content)

#         # Update task status
#         task.status = "completed"
#         task.result = f"Successfully scraped {url}"
#         task.completed_at = timezone.now()  # Ensure completed_at is set
#         task.save()
#     except Exception as e:
#         task.status = "failed"
#         task.result = str(e)
#         task.save()
#         Log.objects.create(log_level="error", message=str(e))
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_website(url, depth, keywords, task):
    visited = set()  # To track visited URLs and avoid duplication

    def scrape_page(page_url, current_depth):
        if page_url in visited or current_depth <= 0:
            return  # Stop recursion if already visited or depth limit reached

        visited.add(page_url)  # Mark the page as visited

        try:
            response = requests.get(page_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Example: Extract title and content
            title = soup.title.string if soup.title else "No title"
            content = soup.get_text()

            # Save scraped data
            ScrapedData.objects.create(
                task=task, url=page_url, title=title, content=content
            )

            # Update task progress (optional)
            task.status = "in_progress"
            task.result = f"Scraped {page_url}"
            task.save()

            # If scraping all pages, follow internal links
            if depth == "all-pages":
                internal_links = [
                    urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True)
                ]
                for link in internal_links:
                    if url in link:  # Ensure link is internal
                        scrape_page(link, current_depth - 1)

        except Exception as e:
            # Handle scraping errors
            task.status = "failed"
            task.result = f"Error scraping {page_url}: {str(e)}"
            task.save()

    # Start scraping from the initial URL
    scrape_page(url, 5)  # Use an arbitrary high depth for "all pages"

    # After finishing all pages
    task.status = "completed"
    task.result = "Scraping completed"
    task.save()


def settings(request):
    if request.method == "POST":
        form = SettingsForm(request.POST)
        if form.is_valid():
            frequency = form.cleaned_data["frequency"]
            user_agent = form.cleaned_data["user_agent"]
            rate_limit = form.cleaned_data["rate_limit"]

            # Save settings to Configuration model
            Configuration.objects.update_or_create(
                key="frequency", defaults={"value": frequency}
            )
            Configuration.objects.update_or_create(
                key="user_agent", defaults={"value": user_agent}
            )
            Configuration.objects.update_or_create(
                key="rate_limit", defaults={"value": rate_limit}
            )

            return redirect("settings")
    else:
        # Pre-fill form with current settings
        frequency = (
            Configuration.objects.filter(key="frequency").first().value or "daily"
        )
        user_agent = (
            Configuration.objects.filter(key="user_agent").first().value
            or "Mozilla/5.0"
        )
        rate_limit = (
            Configuration.objects.filter(key="rate_limit").first().value or "10"
        )
        form = SettingsForm(
            initial={
                "frequency": frequency,
                "user_agent": user_agent,
                "rate_limit": rate_limit,
            }
        )

    return render(request, "settings.html", {"form": form})


import csv
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .models import ScrapedData, Task, AnalysisResult


# View to display the analysis page
def analysis_view(request):
    task = (
        Task.objects.last()
    )  # Get the most recent task, or you can adjust this as needed
    scraped_datasets = ScrapedData.objects.all()

    context = {
        "task": task,
        "scraped_datasets": scraped_datasets,
    }
    return render(request, "analysis.html", context)


# Handle data upload and selection for analysis
def start_analysis(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("data-upload")
        selected_dataset_id = request.POST.get("scraped-data")

        data = None  # Initialize data

        if uploaded_file:
            # Save the uploaded file to the media folder
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.url(filename)

            # You can now read the file and process it (e.g., CSV or JSON data)
            if uploaded_file.name.endswith(".csv"):
                # Process CSV data
                with open(fs.path(filename), "r") as file:
                    reader = csv.reader(file)
                    # Process the rows
                    data = list(reader)
            elif uploaded_file.name.endswith(".json"):
                # Process JSON data
                with open(fs.path(filename), "r") as file:
                    data = json.load(file)

        elif selected_dataset_id != "none":
            # Fetch the selected scraped dataset for analysis
            dataset = ScrapedData.objects.get(id=selected_dataset_id)
            data = dataset.content  # Using dataset.content as the raw data

        # Create a new Task object to track the analysis
        task = Task.objects.create(
            task_type="analysis",
            status="pending",
        )

        # Optionally, store the raw data in AnalysisResult or another model
        AnalysisResult.objects.create(
            task=task,
            analysis_type="text-analysis",
            result_data=str(data),  # Store the data as text
        )

        # Redirect to the analysis options page
        return redirect("analysis_results", task_id=task.id)

    return redirect("analysis")


# Handle text analysis options and perform analysis
import json
from django.shortcuts import redirect
from .models import Task, AnalysisResult


def perform_analysis(request):
    if request.method == "POST":
        task_id = request.POST.get("task_id")
        analysis_types = request.POST.getlist("analysis-type")

        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            return redirect("analysis")  # Redirect if task does not exist

        # Assuming we already have the content for analysis
        if task.scrapeddata_set.exists():
            text_content = task.scrapeddata_set.first().content
        else:
            text_content = "No data found for analysis."

        # Perform the selected analysis types
        results = {}
        if "keyword-extraction" in analysis_types:
            results["keywords"] = perform_keyword_extraction(text_content)
        if "sentiment-analysis" in analysis_types:
            results["sentiment"] = perform_sentiment_analysis(text_content)
        if "content-categorization" in analysis_types:
            results["categories"] = perform_content_categorization(text_content)
        if "entity-recognition" in analysis_types:
            results["entities"] = perform_entity_recognition(text_content)

        # Convert the results to JSON before saving
        try:
            result_json = json.dumps(results)  # Ensure the data is JSON-formatted
        except (TypeError, ValueError) as e:
            result_json = "{}"  # Fallback to empty JSON if conversion fails

        # Update task status and save the result
        task.status = "completed"
        task.save()

        # Save analysis results
        AnalysisResult.objects.update_or_create(
            task=task,
            defaults={
                "analysis_type": "text-analysis",
                "result_data": result_json,  # Save as JSON
            },
        )

        # Redirect to the analysis results page
        return redirect("analysis_results", task_id=task.id)

    return redirect("analysis")


# View to display the analysis results
import json
from django.shortcuts import render, redirect
from .models import Task, AnalysisResult


import ast


import json


def analysis_results(request, task_id):
    try:
        # Fetch the task and its associated results
        task = Task.objects.get(id=task_id)
        analysis_result = AnalysisResult.objects.get(task=task)
        result_data = analysis_result.result_data

        # Try to decode result_data from JSON
        try:
            results = json.loads(result_data)  # Decode from JSON
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON for task {task_id}. Treating result_data as plain text."
            )
            results = {
                "keywords": "N/A",
                "sentiment": "N/A",
                "categories": "N/A",
                "entities": "N/A",
            }  # Fallback values

        # Assuming result_data is stored as JSON
        context = {
            "task": task,
            "results": results,
        }

        return render(request, "analysis_results.html", context)
    except (Task.DoesNotExist, AnalysisResult.DoesNotExist):
        # Handle cases where the task or results are not found
        return redirect("analysis")  # Redirect to analysis page or show an error


def analysis_options(request, task_id):
    # Fetch the task by task_id and render the appropriate template
    task = Task.objects.get(id=task_id)
    return render(request, "analysis_options.html", {"task": task})


# Function to simulate keyword extraction (just a placeholder)
# def perform_keyword_extraction(data):
#     # Process the data to extract keywords
#     return ["hate", "violence", "abuse"]
from keybert import KeyBERT

# Initialize KeyBERT model
kw_model = KeyBERT()


def perform_keyword_extraction(text):
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words="english"
    )
    # Returns a list of keyword tuples (word, score), we return only the words here
    return [word for word, score in keywords]


# Function to simulate sentiment analysis (just a placeholder)
from transformers import pipeline


# Use Hugging Face pipeline for sentiment analysis
def perform_sentiment_analysis(text):
    # Load pre-trained sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")

    # Perform sentiment analysis on the given text
    result = sentiment_analyzer(text)

    # The result will be a list with a dictionary inside
    sentiment = result[0]["label"]  # 'LABEL_0' for negative, 'LABEL_1' for positive

    return sentiment


# Function to simulate content categorization (just a placeholder)
# def perform_content_categorization(data):
#     # Categorize content
#     return ["Harmful"]
from transformers import pipeline

# Load pre-trained classification model for categorization
classifier = pipeline(
    "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def perform_content_categorization(text):
    result = classifier(text)
    category = result[0]["label"]  # Get the category, e.g., "LABEL_0" or "LABEL_1"

    # Example mapping: You can adjust this based on your requirements
    category_mapping = {
        "LABEL_0": "Harmful",
        "LABEL_1": "Neutral",
    }
    return category_mapping.get(result[0]["label"], "Unknown")


# Function to simulate entity recognition (just a placeholder)
# def perform_entity_recognition(data):
#     # Recognize named entities
#     return ["Entity1", "Entity2"]
import spacy

# Load spaCy's English model for NER
nlp = spacy.load("en_core_web_sm")


def perform_entity_recognition(text):
    doc = nlp(text)
    entities = [
        (ent.text, ent.label_) for ent in doc.ents
    ]  # Extract entities and their labels
    return entities  # Return a list of entities


import io
import csv
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone
from .models import Report
from reportlab.pdfgen import canvas


# Utility function to generate a PDF preview
def generate_report_preview(reports_data, report_format):
    if report_format == "pdf":
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)

        # Add some title and content
        p.drawString(100, 750, "Report Preview")
        p.drawString(100, 730, "Website | Content Type | Detected At")

        # Add report data to the PDF
        y_position = 700
        for report in reports_data:
            report_line = f"{report.get('website', 'N/A')} | {report.get('type', 'N/A')} | {report.get('detected_at', 'N/A')}"
            p.drawString(100, y_position, report_line)
            y_position -= 20  # Move down

        p.showPage()
        p.save()
        buffer.seek(0)
        return HttpResponse(buffer, content_type="application/pdf")

    elif report_format == "excel":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Website", "Content Type", "Detected At"])

        for report in reports_data:
            writer.writerow(
                [
                    report.get("website", "N/A"),
                    report.get("type", "N/A"),
                    report.get("detected_at", "N/A"),
                ]
            )

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="report.csv"'
        response.write(output.getvalue())
        return response

    return None


# View function to handle reports page
def reports_view(request):
    if request.method == "POST":
        start_date = request.POST.get("start-date")
        end_date = request.POST.get("end-date")
        category = request.POST.get("category-filter")
        keyword = request.POST.get("keyword-filter")
        sentiment = request.POST.get("sentiment-filter")
        report_format = request.POST.get("report-format")

        # Filtering logic for reports_data
        reports_data = [
            {"website": "example.com", "type": "harmful", "detected_at": "2024-09-10"},
            {"website": "example.org", "type": "neutral", "detected_at": "2024-09-09"},
            # Mock data - replace with actual query results
        ]

        # Generate report preview based on filters and format
        response = generate_report_preview(reports_data, report_format)
        return response

    # Fetch saved reports from the database
    saved_reports = Report.objects.all()

    context = {
        "saved_reports": saved_reports,
    }
    return render(request, "reports.html", context)


# View to delete a report
def delete_report(request, report_id):
    report = Report.objects.get(id=report_id)
    if request.method == "POST":
        report.delete()
        return redirect("reports")  # Replace with your reports URL name

    return render(request, "reports.html")


def save_scraping_settings(request):
    # Logic to save scraping settings
    return HttpResponse('Save settings')

def error_page(request):
    return render(request, "error_page.html")


def help_view(request):
    return render(request, "help.html")


def profile_view(request):
    return render(request, "profile.html")

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import UserProfile


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import UserProfile


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import UserProfile


@login_required
def update_profile(request):
    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        # Handle the case when the user profile does not exist
        user_profile = UserProfile(
            user=request.user
        )  # Create a new UserProfile if it doesn't exist
        user_profile.save()

    if request.method == "POST":
        # Get the form data from the request
        age = request.POST.get("age")
        gender = request.POST.get("gender")
        height = request.POST.get("height")
        weight = request.POST.get("weight")

        # Update user profile fields based on the submitted form data
        user_profile.age = age
        user_profile.gender = gender
        user_profile.height = height
        user_profile.weight = weight
        user_profile.save()  # Save the updated profile

        return redirect("profile")  # Redirect to the profile page or another page

    context = {"user_profile": user_profile}
    return render(request, "profile.html", context)


from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect


def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("profile")  # Redirect to the profile page or any other page
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})
    return render(request, "login.html")

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()


def logout_view(request):
    # Handle logout logic
    return render(request, "logout.html")
import matplotlib.pyplot as plt
import io
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def generate_plot(request):
    # Create a new figure
    fig, ax = plt.subplots()

    # Example data
    categories = ["Hate Speech", "Violence", "Misinformation"]
    values = [50, 30, 20]

    # Create a bar chart
    ax.bar(categories, values)

    # Set chart title and labels
    ax.set_title("Content Issue Distribution")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Occurrences")

    # Save the figure to an in-memory file
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # Return the image as an HTTP response
    return HttpResponse(buf, content_type="image/png")
import io
import matplotlib.pyplot as plt
from django.http import HttpResponse
from .models import Task
from django.db.models import Sum


def scraping_performance_chart_view(request):
    print("Generating simple test chart...")  # Log statement

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9], label="Test Data")  # Simple line chart
    ax.set_title("Test Chart")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return HttpResponse(buf, content_type="image/png")


from django.shortcuts import render
from django.core.exceptions import ObjectDoesNotExist
from .utils import get_scraping_status, get_harmful_content_trends, get_scraping_performance

# def dashboard(request):
#     try:
#         # Fetch scraping status
#         scraping_status = get_scraping_status()
#         scraping_progress = scraping_status.get('progress', 0)
#         scraping_websites = scraping_status.get('websites', 0)
#         scraping_pages = scraping_status.get('pages', 0)

#         # Fetch recent harmful content
#         recent_harmful_content = HarmfulContent.objects.order_by('-detected_at')[:5]

#         # Fetch system health
#         system_health = SystemHealth.objects.latest('timestamp')

#         # Fetch data visualizations
#         harmful_content_trends = get_harmful_content_trends()
#         scraping_performance = get_scraping_performance()

#         # Fetch recent activity
#         recent_activity = ActivityLog.objects.order_by('-timestamp')[:10]

#     except ObjectDoesNotExist as e:
#         # Handle missing data gracefully
#         scraping_progress = 0
#         scraping_websites = 0
#         scraping_pages = 0
#         recent_harmful_content = []
#         system_health = {'cpu_usage': 0, 'memory_usage': 0, 'storage_used': 0}
#         harmful_content_trends = None
#         scraping_performance = None
#         recent_activity = []

#     context = {
#         'scraping_progress': scraping_progress,
#         'scraping_websites': scraping_websites,
#         'scraping_pages': scraping_pages,
#         'recent_harmful_content': recent_harmful_content,
#         'system_health': system_health,
#         'harmful_content_trends': harmful_content_trends,
#         'scraping_performance': scraping_performance,
#         'recent_activity': recent_activity,
#     }

#     return render(request, 'dashboard.html', context)

# from django.shortcuts import render
import io
import base64
import matplotlib.pyplot as plt
from .models import Task
from django.db.models import Avg, Count, Sum


def get_scraping_performance_chart():
    # Query completed scraping tasks and aggregate performance data
    data = Task.objects.filter(task_type="scraping", status="completed").values(
        "id", "pages_scraped", "duration"
    )

    # Extract task IDs, pages scraped, and duration for each task
    task_ids = [str(item["id"]) for item in data]
    pages_scraped = [item["pages_scraped"] for item in data]
    durations = [
        item["duration"].total_seconds() / 60 if item["duration"] else 0
        for item in data
    ]  # Convert to minutes

    # Plot scraping performance - pages scraped and task duration
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.bar(task_ids, pages_scraped, color="blue", label="Pages Scraped")
    ax1.set_xlabel("Task ID")
    ax1.set_ylabel("Pages Scraped", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot duration on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(task_ids, durations, color="red", label="Duration (mins)")
    ax2.set_ylabel("Duration (minutes)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()

    # Save the figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Encode the image to base64 for embedding in HTML
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


from django.shortcuts import render
from .models import Task, HarmfulContent
from .utils import get_scraping_performance_chart, get_scraping_performance_chart
from django.db.models import Count, Sum, Avg


def dashboard(request):
    # Scraping progress: calculate progress dynamically
    total_websites = Task.objects.filter(task_type="scraping").count()
    completed_websites = Task.objects.filter(
        task_type="scraping", status="completed"
    ).count()
    scraping_progress = (
        (completed_websites / total_websites * 100) if total_websites > 0 else 0
    )

    # Pages processed: sum of all pages scraped
    pages_processed = (
        Task.objects.filter(task_type="scraping", status="completed").aggregate(
            Sum("pages_scraped")
        )["pages_scraped__sum"]
        or 0
    )

    # Harmful content: pull data from HarmfulContent model
    harmful_content = HarmfulContent.objects.values("website", "type")[
        :3
    ]  # Limit to 3 recent harmful content items

    # System health: dummy data, but you can integrate real monitoring
    system_health = {
        "cpu_usage": 45,  # Placeholder value, integrate actual system monitoring
        "memory_usage": 68,  # Placeholder value
        "storage_usage": 80,  # Placeholder value
    }

    # Recent activities: log the latest task and analysis activities
    recent_activities = Task.objects.order_by("-started_at").values(
        "task_type", "result"
    )[:4]

    context = {
        "scraping_progress": scraping_progress,
        "total_websites": total_websites,
        "pages_processed": pages_processed,
        "scraping_performance_chart": get_scraping_performance_chart(),
        "harmful_content": harmful_content,
        "system_health": system_health,
        "recent_activities": [
            {"description": f"{activity['task_type']} task: {activity['result']}"}
            for activity in recent_activities
        ],
    }

    return render(request, "dashboard.html", context)


def pause_task(request, task_id):
    # Logic to pause the task
    task = Task.objects.get(id=task_id)
    task.status = "paused"  # Add 'paused' status to your choices if needed
    task.save()
    return redirect("ongoing_scraping_tasks")


# def pause_task(request, task_id):
#     task = get_object_or_404(Task, id=task_id)
#     if task.status == "in_progress":
#         task.status = "paused"  # Assuming you have a paused status or similar logic
#         task.save()
#     return redirect("ongoing_scraping_tasks")


def cancel_task(request, task_id):
    # Logic to cancel the task
    task = Task.objects.get(id=task_id)
    task.status = "cancelled"  
    return redirect("ongoing_scraping_tasks")
