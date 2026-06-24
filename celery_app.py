"""
Virtual Chemistry Lab API - Celery Application
Celery configuration for background task processing.
"""

from celery import Celery
from app.config import settings

celery = Celery(__name__)

celery.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600 * 24,  # Results expire after 24 hours
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": 43200,  # 12 hours
    },
)

# Auto-discover tasks
celery.autodiscover_tasks(["app.workers.tasks"])
