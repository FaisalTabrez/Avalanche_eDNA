"""
API module for eDNA analysis system.

This module provides REST API endpoints for integrating the report management
and analysis system with external applications.
"""

from .report_management_api import app

__all__ = [
    'app'
]