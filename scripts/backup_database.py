#!/usr/bin/env python3
"""
Database backup and maintenance script for eDNA analysis system.

This script provides automated backup, restoration, and maintenance
operations for the database.
"""

import os
import sys
import shutil
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.database.config import get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseBackupManager:
    """Manages database backups and maintenance operations."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.backup_dir = project_root / "data" / "report_storage" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a database backup."""

        if self.config.db_type != "sqlite":
            raise NotImplementedError("Backup only implemented for SQLite")

        db_path = Path(self.config.sqlite_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        # Generate backup filename
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"reports_backup_{timestamp}.db"

        backup_path = self.backup_dir / backup_name

        # Create backup by copying the database file
        shutil.copy2(db_path, backup_path)

        # Verify backup integrity
        if self._verify_backup(backup_path):
            logger.info(f"✓ Database backup created: {backup_path}")
            return str(backup_path)
        else:
            backup_path.unlink()  # Delete corrupted backup
            raise RuntimeError("Backup verification failed")

    def restore_backup(
        self, backup_path: str, target_path: Optional[str] = None
    ) -> bool:
        """Restore database from backup."""

        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        target_path = target_path or self.config.sqlite_path
        target_file = Path(target_path)

        # Create backup of current database before restoration
        if target_file.exists():
            recovery_backup = target_file.with_suffix(".recovery.db")
            shutil.copy2(target_file, recovery_backup)
            logger.info(f"Created recovery backup: {recovery_backup}")

        # Restore from backup
        shutil.copy2(backup_file, target_file)

        # Verify restoration
        if self._verify_backup(target_file):
            logger.info(f"✓ Database restored from: {backup_path}")
            return True
        else:
            # Restore from recovery backup if verification fails
            if recovery_backup.exists():
                shutil.copy2(recovery_backup, target_file)
                logger.warning("Restoration failed, recovered from recovery backup")
            return False

    def list_backups(self) -> List[Path]:
        """List all available backups."""
        return sorted(
            self.backup_dir.glob("*.db"), key=lambda x: x.stat().st_mtime, reverse=True
        )

    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Remove backups older than retention period."""

        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0

        for backup_file in self.backup_dir.glob("*.db"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                removed_count += 1
                logger.info(f"Removed old backup: {backup_file.name}")

        logger.info(f"Cleaned up {removed_count} old backups")
        return removed_count

    def get_backup_info(self) -> dict:
        """Get information about backups."""

        backups = self.list_backups()
        total_size = sum(b.stat().st_size for b in backups)

        return {
            "total_backups": len(backups),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_backup": backups[-1].name if backups else None,
            "newest_backup": backups[0].name if backups else None,
            "backup_directory": str(self.backup_dir),
        }

    def _verify_backup(self, db_path: Path) -> bool:
        """Verify database backup integrity."""

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check if we can query the schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            # Verify expected tables exist
            expected_tables = {
                "organism_profiles",
                "datasets",
                "analysis_reports",
                "sequences",
                "taxonomic_assignments",
                "clustering_results",
                "novelty_detections",
                "similarity_matrices",
                "report_comparisons",
            }

            found_tables = {row[0] for row in tables}
            missing_tables = expected_tables - found_tables

            conn.close()

            if missing_tables:
                logger.error(f"Backup missing tables: {missing_tables}")
                return False

            return True

        except Exception as e:
            logger.error(f"Backup verification failed: {str(e)}")
            return False


def main():
    """Main CLI interface for backup operations."""

    import argparse

    parser = argparse.ArgumentParser(description="Database backup and maintenance")
    parser.add_argument(
        "action", choices=["backup", "restore", "list", "cleanup", "info"]
    )
    parser.add_argument("--backup-file", help="Backup file path for restore")
    parser.add_argument(
        "--retention-days", type=int, default=30, help="Days to retain backups"
    )
    parser.add_argument("--target-path", help="Target path for restoration")

    args = parser.parse_args()

    backup_manager = DatabaseBackupManager()

    try:
        if args.action == "backup":
            backup_path = backup_manager.create_backup()
            print(f"Backup created: {backup_path}")

        elif args.action == "restore":
            if not args.backup_file:
                print("Error: --backup-file required for restore")
                return 1

            success = backup_manager.restore_backup(args.backup_file, args.target_path)
            print(f"Restore {'successful' if success else 'failed'}")

        elif args.action == "list":
            backups = backup_manager.list_backups()
            if backups:
                print("Available backups:")
                for backup in backups:
                    size_mb = backup.stat().st_size / (1024 * 1024)
                    mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                    print(".2f")
            else:
                print("No backups found")

        elif args.action == "cleanup":
            removed = backup_manager.cleanup_old_backups(args.retention_days)
            print(f"Removed {removed} old backups")

        elif args.action == "info":
            info = backup_manager.get_backup_info()
            print("Backup Information:")
            print(f"  Total backups: {info['total_backups']}")
            print(".2f")
            print(f"  Oldest: {info['oldest_backup'] or 'None'}")
            print(f"  Newest: {info['newest_backup'] or 'None'}")
            print(f"  Directory: {info['backup_directory']}")

    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
