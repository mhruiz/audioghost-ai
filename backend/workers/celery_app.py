"""
Celery Application Configuration
"""
from celery import Celery

import os

celery_app = Celery(
    "audioghost",
    broker=os.getenv("REDIS_URL", "redis://redis_docker:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis_docker:6379/0"),
    include=["workers.tasks"]
)

# Celery Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time (GPU memory)
    result_expires=86400,  # Results expire after 24 hours
)
