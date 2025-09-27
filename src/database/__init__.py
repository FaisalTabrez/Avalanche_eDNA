"""
Database module for eDNA analysis report management and organism profiling.

This module provides database connectivity, schema management, and data access
layers for storing and retrieving analysis reports, organism profiles, and
cross-analysis results.
"""

from .schema import DatabaseSchema
from .models import (
    OrganismProfile, AnalysisReport, DatasetInfo, 
    SimilarityMatrix, ReportComparison
)
from .manager import DatabaseManager
from .queries import ReportQueryEngine

__all__ = [
    'DatabaseSchema',
    'OrganismProfile', 
    'AnalysisReport',
    'DatasetInfo',
    'SimilarityMatrix',
    'ReportComparison',
    'DatabaseManager',
    'ReportQueryEngine'
]