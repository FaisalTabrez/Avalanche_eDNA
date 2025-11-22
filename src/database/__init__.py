"""
Database module for eDNA analysis report management and organism profiling.

This module provides database connectivity, schema management, and data access
layers for storing and retrieving analysis reports, organism profiles, and
cross-analysis results.

Supports both SQLite (development) and PostgreSQL (production) databases.
"""

from .schema import DatabaseSchema
from .models import (
    OrganismProfile, AnalysisReport, DatasetInfo, 
    SimilarityMatrix, ReportComparison
)
from .manager import DatabaseManager
from .queries import ReportQueryEngine
from .connection import (
    DatabaseConnection, 
    DatabaseConfig, 
    get_database_connection,
    reset_database_connection
)
from .migration import DatabaseMigrator, migrate_database
from .sql_schemas import get_schema_for_database, get_indexes_for_database

__all__ = [
    'DatabaseSchema',
    'OrganismProfile', 
    'AnalysisReport',
    'DatasetInfo',
    'SimilarityMatrix',
    'ReportComparison',
    'DatabaseManager',
    'ReportQueryEngine',
    'DatabaseConnection',
    'DatabaseConfig',
    'get_database_connection',
    'reset_database_connection',
    'DatabaseMigrator',
    'migrate_database',
    'get_schema_for_database',
    'get_indexes_for_database'
]