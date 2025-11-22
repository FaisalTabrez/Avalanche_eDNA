"""
Task queue system using Celery

This module provides a distributed task queue for background processing
of long-running operations like analysis, training, and data downloads.
"""

from .celery_app import celery_app

__all__ = ['celery_app']
