from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),  # Dashboard
    path(
        "start-task/", views.start_scraping_task, name="start_scraping_task"
    ),  # Start scraping task
    path("pause-task/<int:task_id>/", views.pause_task, name="pause_task"),
    path("cancel-task/<int:task_id>/", views.cancel_task, name="cancel_task"),
    path("reports/", views.reports_view, name="reports"),
    path("delete-report/<int:report_id>/", views.delete_report, name="delete_report"),
    path(
        "ongoing-tasks/", views.ongoing_scraping_tasks, name="ongoing_scraping_tasks"
    ),  # Ongoing tasks
    path(
        "completed-tasks/",
        views.completed_scraping_tasks,
        name="completed_scraping_tasks",
    ),  # Completed tasks
    path("download-data/<int:task_id>/", views.download_data, name="download_data"),
    path("view-results/<int:task_id>/", views.view_results, name="view_results"),
    path("settings/", views.settings, name="settings"),  # Settings page
    path(
        "save-settings/", views.save_scraping_settings, name="save_scraping_settings"
    ),  # Save settings
    path("generate-plot/", views.generate_plot, name="generate_plot"),
    path(
        "scraping-performance-chart/",
        views.scraping_performance_chart_view,
        name="scraping_performance_chart",
    ),
    path("analysis/", views.analysis_view, name="analysis"),
    path("start-analysis/", views.start_analysis, name="start_analysis"),
    path("perform-analysis/", views.perform_analysis, name="perform_analysis"),
    path(
        "analysis-results/<int:task_id>/",
        views.analysis_results,
        name="analysis_results",
    ),
    path(
        "analysis-options/<int:task_id>/",
        views.analysis_options,
        name="analysis_options",
    ),
    path("error/", views.error_page, name="error_page"),
    path("help/", views.help_view, name="help"),  # Help page
    path("profile/", views.profile_view, name="profile"),  # Profile page
    path("update-profile/", views.update_profile, name="update_profile"),
    path(
        "accounts/login/", views.login_view, name="login"
    ), 
    path("logout/", views.logout_view, name="logout"),  # Logout
]
