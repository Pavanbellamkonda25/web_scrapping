# utils.py
import matplotlib.pyplot as plt
import io
import base64
from .models import HarmfulContent, ActivityLog, SystemHealth


def get_scraping_status():
    # Placeholder function to get scraping status
    return {
        "progress": 70,
        "websites": 3,
        "pages": 120,
    }
from django.db import models

def get_harmful_content_trends():
    # Generate a simple bar chart as a placeholder
    data = HarmfulContent.objects.values("type").annotate(count=models.Count("type"))
    types = [item["type"] for item in data]
    counts = [item["count"] for item in data]

    plt.figure(figsize=(6, 4))
    plt.bar(types, counts, color="red")
    plt.xlabel("Content Type")
    plt.ylabel("Count")
    plt.title("Harmful Content Detection Trends")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def get_scraping_performance():
    # Generate a simple line chart as a placeholder
    data = ActivityLog.objects.filter(description__icontains="Scraping task").order_by(
        "timestamp"
    )
    timestamps = [activity.timestamp for activity in data]
    counts = range(1, len(timestamps) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(timestamps, counts, marker="o")
    plt.xlabel("Timestamp")
    plt.ylabel("Tasks Completed")
    plt.title("Scraping Task Performance")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

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
