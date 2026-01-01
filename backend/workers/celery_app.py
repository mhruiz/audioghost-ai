"""
Celery Application Configuration
"""
from celery import Celery
from celery.signals import worker_ready

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


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Auto-preload model when worker starts if PRELOAD_MODEL is set"""
    preload_model = os.getenv("PRELOAD_MODEL", "").strip()
    
    if preload_model and preload_model in ["small", "base", "large"]:
        print(f"[STARTUP] PRELOAD_MODEL={preload_model} detected. Scheduling preload...")
        
        # Import here to avoid circular imports
        from workers.tasks import preload_model_task
        
        # Delay slightly to ensure worker is fully ready
        preload_model_task.apply_async(
            args=[preload_model, False],
            countdown=2  # Wait 2 seconds before starting
        )
    elif preload_model:
        print(f"[STARTUP] Invalid PRELOAD_MODEL value: {preload_model}. Use: small, base, large")
    else:
        print("[STARTUP] No PRELOAD_MODEL configured. Models will load on first request.")

