# Generated by Django 5.1 on 2024-09-10 11:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0004_activitylog_harmfulcontent_systemhealth_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Report',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('generated_at', models.DateTimeField(auto_now_add=True)),
                ('file_url', models.URLField()),
                ('filters', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
