"""
Database configuration for eDNA analysis system.

This module provides database connection settings and configuration
for different environments (development, production, testing).
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    # Database type: 'sqlite', 'postgresql', 'mysql'
    db_type: str = "sqlite"

    # SQLite settings
    sqlite_path: str = "data/reports.db"

    # PostgreSQL/MySQL settings
    host: str = "localhost"
    port: int = 5432  # 3306 for MySQL
    database: str = "edna_reports"
    username: str = "edna_user"
    password: str = ""

    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30

    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        if self.db_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == "mysql":
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


def get_config(env: str = None) -> DatabaseConfig:
    """Get database configuration for environment."""

    if env is None:
        env = os.getenv("EDNA_ENV", "development")

    base_config = DatabaseConfig()

    if env == "development":
        # SQLite for development
        base_config.db_type = "sqlite"
        base_config.sqlite_path = "data/reports.db"

    elif env == "testing":
        # Separate SQLite database for testing
        base_config.db_type = "sqlite"
        base_config.sqlite_path = "data/test_reports.db"

    elif env == "production":
        # PostgreSQL for production
        base_config.db_type = os.getenv("DB_TYPE", "postgresql")
        base_config.host = os.getenv("DB_HOST", "localhost")
        base_config.port = int(os.getenv("DB_PORT", "5432"))
        base_config.database = os.getenv("DB_NAME", "edna_reports")
        base_config.username = os.getenv("DB_USER", "edna_user")
        base_config.password = os.getenv("DB_PASSWORD", "")

        # Production pool settings
        base_config.pool_size = 10
        base_config.max_overflow = 20

    return base_config


# Environment-specific configurations
CONFIGS: Dict[str, DatabaseConfig] = {
    "development": get_config("development"),
    "testing": get_config("testing"),
    "production": get_config("production"),
}
