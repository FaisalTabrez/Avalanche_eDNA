#!/usr/bin/env python3
"""
Database monitoring and health check script for eDNA analysis system.

This script provides monitoring, health checks, and performance analysis
for the database.
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Monitors database health and performance."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive database health status."""

        health = {
            "timestamp": datetime.now().isoformat(),
            "database_exists": self.db_manager.schema.database_exists(),
            "schema_version": self.db_manager.schema.get_schema_version(),
            "connection_status": "unknown",
            "performance_metrics": {},
            "data_integrity": {},
            "storage_metrics": {},
            "recommendations": [],
        }

        if not health["database_exists"]:
            health["connection_status"] = "database_not_found"
            health["recommendations"].append(
                "Run setup_report_management.py to initialize database"
            )
            return health

        try:
            # Test connection
            with self.db_manager.get_connection() as conn:
                health["connection_status"] = "healthy"
                health["performance_metrics"] = self._get_performance_metrics(conn)
                health["data_integrity"] = self._check_data_integrity(conn)
                health["storage_metrics"] = self._get_storage_metrics()

        except Exception as e:
            health["connection_status"] = f"error: {str(e)}"
            health["recommendations"].append(f"Database connection failed: {str(e)}")

        # Generate recommendations
        health["recommendations"].extend(self._generate_recommendations(health))

        return health

    def _get_performance_metrics(self, conn) -> Dict[str, Any]:
        """Get database performance metrics."""

        metrics = {}

        try:
            # Query execution time for common operations
            start_time = time.time()
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_reports")
            metrics["total_reports"] = cursor.fetchone()[0]
            query_time = time.time() - start_time
            metrics["query_time_seconds"] = query_time

            # Table sizes
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            table_sizes = {}
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                table_sizes[table] = cursor.fetchone()[0]

            metrics["table_sizes"] = table_sizes

            # Index usage (SQLite specific)
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='index'
            """)
            metrics["total_indexes"] = len(cursor.fetchall())

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    def _check_data_integrity(self, conn) -> Dict[str, Any]:
        """Check data integrity constraints."""

        integrity = {"passed": 0, "failed": 0, "checks": []}

        checks = [
            {
                "name": "foreign_key_constraints",
                "query": "PRAGMA foreign_key_check",
                "expected_empty": True,
            },
            {
                "name": "unique_constraints",
                "query": "SELECT name FROM sqlite_master WHERE type='table'",
                "custom_check": self._check_unique_constraints,
            },
            {
                "name": "organism_id_format",
                "query": "SELECT organism_id FROM organism_profiles LIMIT 100",
                "custom_check": self._check_organism_id_format,
            },
        ]

        for check in checks:
            try:
                if "custom_check" in check:
                    result = check["custom_check"](conn)
                else:
                    cursor = conn.execute(check["query"])
                    results = cursor.fetchall()
                    result = len(results) == 0 if check.get("expected_empty") else True

                status = "passed" if result else "failed"
                integrity[status] += 1
                integrity["checks"].append(
                    {
                        "name": check["name"],
                        "status": status,
                        "details": f"Check completed successfully"
                        if result
                        else "Integrity violation detected",
                    }
                )

            except Exception as e:
                integrity["failed"] += 1
                integrity["checks"].append(
                    {"name": check["name"], "status": "error", "details": str(e)}
                )

        return integrity

    def _get_storage_metrics(self) -> Dict[str, Any]:
        """Get database storage and file system metrics."""

        db_path = Path(str(self.db_manager.db_path))

        if not db_path.exists():
            return {"error": "database_file_not_found"}

        # File system metrics
        stat = db_path.stat()
        file_size_mb = stat.st_size / (1024 * 1024)

        # Get disk usage
        disk_usage = psutil.disk_usage(str(db_path.parent))

        return {
            "database_file_size_mb": round(file_size_mb, 2),
            "disk_free_mb": round(disk_usage.free / (1024 * 1024), 2),
            "disk_total_mb": round(disk_usage.total / (1024 * 1024), 2),
            "disk_usage_percent": disk_usage.percent,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def _check_unique_constraints(self, conn) -> bool:
        """Check for unique constraint violations."""
        try:
            # Check organism_id uniqueness
            cursor = conn.execute("""
                SELECT organism_id, COUNT(*) as count
                FROM organism_profiles
                GROUP BY organism_id
                HAVING count > 1
            """)
            if cursor.fetchone():
                return False

            # Check report_id uniqueness
            cursor = conn.execute("""
                SELECT report_id, COUNT(*) as count
                FROM analysis_reports
                GROUP BY report_id
                HAVING count > 1
            """)
            if cursor.fetchone():
                return False

            return True

        except Exception:
            return False

    def _check_organism_id_format(self, conn) -> bool:
        """Check organism ID format consistency."""
        try:
            cursor = conn.execute("""
                SELECT organism_id FROM organism_profiles
                WHERE organism_id NOT LIKE 'ORG_%'
            """)
            invalid_ids = cursor.fetchall()
            return len(invalid_ids) == 0

        except Exception:
            return False

    def _generate_recommendations(self, health: Dict[str, Any]) -> List[str]:
        """Generate maintenance recommendations based on health status."""

        recommendations = []

        if health["connection_status"] != "healthy":
            recommendations.append("Fix database connection issues before proceeding")

        # Storage recommendations
        storage = health.get("storage_metrics", {})
        if storage.get("disk_usage_percent", 0) > 90:
            recommendations.append("Disk space is low - consider cleanup or expansion")

        db_size = storage.get("database_file_size_mb", 0)
        if db_size > 1000:  # > 1GB
            recommendations.append("Database is large - consider archiving old data")

        # Performance recommendations
        perf = health.get("performance_metrics", {})
        query_time = perf.get("query_time_seconds", 0)
        if query_time > 1.0:
            recommendations.append(
                "Query performance is slow - consider adding indexes"
            )

        # Data integrity recommendations
        integrity = health.get("data_integrity", {})
        if integrity.get("failed", 0) > 0:
            recommendations.append(
                "Data integrity issues detected - run integrity check"
            )

        # Table size recommendations
        table_sizes = perf.get("table_sizes", {})
        large_tables = [table for table, size in table_sizes.items() if size > 100000]
        if large_tables:
            recommendations.append(
                f"Large tables detected: {', '.join(large_tables)} - consider partitioning"
            )

        return recommendations

    def run_maintenance(self) -> Dict[str, Any]:
        """Run database maintenance operations."""

        maintenance = {
            "timestamp": datetime.now().isoformat(),
            "operations": [],
            "success": True,
        }

        try:
            with self.db_manager.get_connection() as conn:
                # Vacuum database (SQLite optimization)
                conn.execute("VACUUM")
                maintenance["operations"].append(
                    {
                        "name": "vacuum",
                        "status": "completed",
                        "description": "Database optimization completed",
                    }
                )

                # Analyze query patterns
                conn.execute("ANALYZE")
                maintenance["operations"].append(
                    {
                        "name": "analyze",
                        "status": "completed",
                        "description": "Query statistics updated",
                    }
                )

        except Exception as e:
            maintenance["success"] = False
            maintenance["error"] = str(e)

        return maintenance


def main():
    """CLI interface for database monitoring."""

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Database monitoring and health checks"
    )
    parser.add_argument("action", choices=["health", "maintenance", "monitor"])
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument(
        "--interval", type=int, default=60, help="Monitoring interval in seconds"
    )

    args = parser.parse_args()

    monitor = DatabaseMonitor()

    try:
        if args.action == "health":
            health = monitor.get_health_status()

            if args.output == "json":
                print(json.dumps(health, indent=2))
            else:
                print("Database Health Status")
                print("=" * 50)
                print(f"Status: {health['connection_status']}")
                print(f"Schema Version: {health['schema_version']}")
                print(f"Database Exists: {health['database_exists']}")

                if "performance_metrics" in health:
                    perf = health["performance_metrics"]
                    print(f"\nPerformance Metrics:")
                    print(f"  Total Reports: {perf.get('total_reports', 'N/A')}")
                    print(".3f")
                    print(f"  Total Indexes: {perf.get('total_indexes', 'N/A')}")

                if "storage_metrics" in health:
                    storage = health["storage_metrics"]
                    print(f"\nStorage Metrics:")
                    print(".2f")
                    print(".1f")

                if health["recommendations"]:
                    print(f"\nRecommendations:")
                    for rec in health["recommendations"]:
                        print(f"  • {rec}")

        elif args.action == "maintenance":
            result = monitor.run_maintenance()

            if args.output == "json":
                print(json.dumps(result, indent=2))
            else:
                print("Database Maintenance Results")
                print("=" * 50)
                print(f"Success: {result['success']}")

                for op in result["operations"]:
                    status_icon = "✓" if op["status"] == "completed" else "✗"
                    print(f"{status_icon} {op['name']}: {op['description']}")

        elif args.action == "monitor":
            print("Starting continuous monitoring (Ctrl+C to stop)...")
            try:
                while True:
                    health = monitor.get_health_status()
                    status = "✓" if health["connection_status"] == "healthy" else "✗"
                    print(
                        f"{datetime.now().strftime('%H:%M:%S')} {status} {health['connection_status']}"
                    )
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped")

    except Exception as e:
        logger.error(f"Monitoring operation failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
