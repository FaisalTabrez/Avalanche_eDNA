#!/usr/bin/env python3
"""
Database migration script for eDNA analysis system.

This script handles database schema migrations and version upgrades.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from database.schema import DatabaseSchema
from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Handles database schema migrations."""

    def __init__(self, db_path: str = None):
        self.schema = DatabaseSchema(db_path)
        self.db_path = self.schema.db_path

    def get_current_version(self) -> str:
        """Get current database schema version."""
        return self.schema.get_schema_version() or "0.0.0"

    def migrate_to_version(self, target_version: str) -> bool:
        """Migrate database to target version."""

        current_version = self.get_current_version()
        logger.info(f"Migrating from {current_version} to {target_version}")

        # Define migration path
        migrations = self._get_migration_path(current_version, target_version)

        if not migrations:
            logger.info("No migrations needed")
            return True

        # Execute migrations
        for migration in migrations:
            logger.info(f"Executing migration: {migration['name']}")
            try:
                migration["function"]()
                self._update_schema_version(migration["version"])
                logger.info(f"✓ Migration {migration['name']} completed")
            except Exception as e:
                logger.error(f"✗ Migration {migration['name']} failed: {str(e)}")
                return False

        logger.info(f"✓ Migration to {target_version} completed successfully")
        return True

    def _get_migration_path(
        self, from_version: str, to_version: str
    ) -> List[Dict[str, Any]]:
        """Get the sequence of migrations needed."""

        # Define available migrations
        migrations = [
            {
                "version": "1.0.0",
                "name": "initial_schema",
                "function": self._migrate_to_1_0_0,
                "requires": "0.0.0",
            },
            {
                "version": "1.1.0",
                "name": "add_performance_indexes",
                "function": self._migrate_to_1_1_0,
                "requires": "1.0.0",
            },
            {
                "version": "1.2.0",
                "name": "add_environmental_metadata",
                "function": self._migrate_to_1_2_0,
                "requires": "1.1.0",
            },
        ]

        # Find migration path
        migration_path = []
        current = from_version

        while current != to_version:
            next_migration = None
            for migration in migrations:
                if migration["requires"] == current:
                    next_migration = migration
                    break

            if next_migration is None:
                logger.error(f"No migration path from {current} to {to_version}")
                return []

            migration_path.append(next_migration)
            current = next_migration["version"]

        return migration_path

    def _migrate_to_1_0_0(self):
        """Initial schema creation."""
        self.schema.create_database()

    def _migrate_to_1_1_0(self):
        """Add performance indexes."""
        with self.schema.conn.cursor() as cursor:
            # Additional indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_sequences_gc_content ON sequences (gc_content)",
                "CREATE INDEX IF NOT EXISTS idx_sequences_quality ON sequences (quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_taxonomy_method ON taxonomic_assignments (assignment_method)",
                "CREATE INDEX IF NOT EXISTS idx_clustering_outliers ON clustering_results (is_outlier)",
                "CREATE INDEX IF NOT EXISTS idx_novelty_validation ON novelty_detections (validation_status)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

    def _migrate_to_1_2_0(self):
        """Add environmental metadata columns."""
        with self.schema.conn.cursor() as cursor:
            # Add new columns to datasets table
            alterations = [
                "ALTER TABLE datasets ADD COLUMN salinity_ppt REAL",
                "ALTER TABLE datasets ADD COLUMN dissolved_oxygen REAL",
                "ALTER TABLE datasets ADD COLUMN turbidity_ntu REAL",
                "ALTER TABLE datasets ADD COLUMN sampling_method TEXT",
                "ALTER TABLE datasets ADD COLUMN sample_volume_ml REAL",
            ]

            for alter_sql in alterations:
                try:
                    cursor.execute(alter_sql)
                except sqlite3.OperationalError:
                    # Column might already exist
                    pass

    def _update_schema_version(self, version: str):
        """Update schema version in database."""
        with self.schema.conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO system_metadata (key, value, data_type, description)
                VALUES (?, ?, ?, ?)
            """,
                ("schema_version", version, "string", "Database schema version"),
            )

    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        current_version = self.get_current_version()
        latest_version = "1.2.0"  # Update this as new migrations are added

        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "needs_migration": current_version != latest_version,
            "migration_available": current_version < latest_version,
        }


def main():
    """CLI interface for migrations."""

    import argparse

    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument("action", choices=["status", "migrate", "upgrade"])
    parser.add_argument("--target-version", help="Target version for migration")
    parser.add_argument("--db-path", help="Database file path")

    args = parser.parse_args()

    migration = DatabaseMigration(args.db_path)

    try:
        if args.action == "status":
            status = migration.get_migration_status()
            print("Migration Status:")
            print(f"  Current version: {status['current_version']}")
            print(f"  Latest version: {status['latest_version']}")
            print(f"  Needs migration: {status['needs_migration']}")

        elif args.action == "migrate":
            if not args.target_version:
                print("Error: --target-version required for migrate")
                return 1

            success = migration.migrate_to_version(args.target_version)
            print(f"Migration {'successful' if success else 'failed'}")

        elif args.action == "upgrade":
            status = migration.get_migration_status()
            if status["migration_available"]:
                success = migration.migrate_to_version(status["latest_version"])
                print(f"Upgrade {'successful' if success else 'failed'}")
            else:
                print("Database is already up to date")

    except Exception as e:
        logger.error(f"Migration operation failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
