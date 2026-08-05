"""
Database module for eDNA analysis report management and organism profiling.

This module provides database connectivity, schema management, and data access
layers for storing and retrieving analysis reports, organism profiles, and
cross-analysis results.
"""

from .manager import DatabaseManager
from .models import (
    AnalysisReport,
    DatasetInfo,
    OrganismProfile,
    ReportComparison,
    SimilarityMatrix,
)
from .queries import ReportQueryEngine
from .schema import DatabaseSchema

__all__ = [
    "DatabaseSchema",
    "OrganismProfile",
    "AnalysisReport",
    "DatasetInfo",
    "SimilarityMatrix",
    "ReportComparison",
    "DatabaseManager",
    "ReportQueryEngine",
]
