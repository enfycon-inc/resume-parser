from celery import Celery
import os

# Celery Configuration
# Broker: Where tasks are sent (Redis)
# Backend: Where results are stored (Redis)
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0")

celery_app = Celery(
    "ats_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks"] # Tell celery where to find the tasks
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300, # 5 minutes max for one resume
    broker_connection_retry_on_startup=True,
    redis_backend_health_check_interval=30,
)
