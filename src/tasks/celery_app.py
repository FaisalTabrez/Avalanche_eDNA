"""
Celery application configuration

This module configures the Celery application for distributed task processing.
"""

import os
from celery import Celery
from celery.schedules import crontab
from kombu import Queue


# Get configuration from environment
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)


# Create Celery application
celery_app = Celery(
    'avalanche_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'src.tasks.analysis_tasks',
        'src.tasks.training_tasks',
        'src.tasks.download_tasks',
        'src.tasks.maintenance_tasks',
    ]
)


# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_persistent=True,
    result_extended=True,  # Store additional task metadata
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Disable prefetching for fair distribution
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks
    worker_disable_rate_limits=False,
    
    # Broker settings
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    
    # Task routing
    task_routes={
        'src.tasks.analysis_tasks.*': {'queue': 'analysis'},
        'src.tasks.training_tasks.*': {'queue': 'training'},
        'src.tasks.download_tasks.*': {'queue': 'downloads'},
        'src.tasks.maintenance_tasks.*': {'queue': 'maintenance'},
    },
    
    # Task queues
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('analysis', routing_key='analysis'),
        Queue('training', routing_key='training'),
        Queue('downloads', routing_key='downloads'),
        Queue('maintenance', routing_key='maintenance'),
    ),
    
    # Default queue
    task_default_queue='default',
    task_default_exchange='tasks',
    task_default_routing_key='default',
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-results': {
            'task': 'src.tasks.maintenance_tasks.cleanup_old_results',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        'cleanup-temp-files': {
            'task': 'src.tasks.maintenance_tasks.cleanup_temp_files',
            'schedule': crontab(hour='*/6'),  # Every 6 hours
        },
        'backup-database': {
            'task': 'src.tasks.maintenance_tasks.backup_database',
            'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
        },
        'monitor-system-health': {
            'task': 'src.tasks.maintenance_tasks.monitor_system_health',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes
        },
    },
)


# Task state tracking
class TaskState:
    """Constants for task states"""
    PENDING = 'PENDING'
    STARTED = 'STARTED'
    RETRY = 'RETRY'
    FAILURE = 'FAILURE'
    SUCCESS = 'SUCCESS'
    REVOKED = 'REVOKED'


# Custom task base class with progress tracking
from celery import Task


class ProgressTask(Task):
    """Base task class with progress tracking"""
    
    def update_progress(self, current, total, status='PROGRESS'):
        """
        Update task progress
        
        Args:
            current: Current progress value
            total: Total progress value
            status: Status message
        """
        self.update_state(
            state=status,
            meta={
                'current': current,
                'total': total,
                'percent': int((current / total) * 100) if total > 0 else 0
            }
        )


# Register custom task base
celery_app.Task = ProgressTask


if __name__ == '__main__':
    celery_app.start()
