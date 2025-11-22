#!/usr/bin/env python3
"""
Database Migration Script - SQLite to PostgreSQL

This script migrates data from SQLite to PostgreSQL for production deployment.

Usage:
    python scripts/migrate_to_postgres.py --validate-only
    python scripts/migrate_to_postgres.py --migrate
    python scripts/migrate_to_postgres.py --setup-schema
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseConfig, DatabaseConnection
from src.database.migration import DatabaseMigrator, migrate_database
from src.database.sql_schemas import get_schema_for_database, get_indexes_for_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_postgres_schema():
    """Create PostgreSQL schema and indexes"""
    logger.info("Setting up PostgreSQL schema")
    
    config = DatabaseConfig()
    
    if not config.is_postgres:
        logger.error("DB_TYPE must be set to 'postgresql' in environment")
        return False
    
    db_conn = DatabaseConnection(config)
    
    try:
        with db_conn.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables
            schema = get_schema_for_database('postgresql')
            for table_name, create_sql in schema.items():
                logger.info(f"Creating table: {table_name}")
                cursor.execute(create_sql)
            
            # Create indexes
            indexes = get_indexes_for_database('postgresql')
            for index_sql in indexes:
                logger.info(f"Creating index: {index_sql[:50]}...")
                cursor.execute(index_sql)
            
            conn.commit()
        
        logger.info("PostgreSQL schema created successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL schema: {e}")
        return False


def validate_migration():
    """Validate existing migration"""
    logger.info("Validating migration")
    
    config = DatabaseConfig()
    
    pg_config = {
        'host': config.pg_host,
        'port': config.pg_port,
        'database': config.pg_database,
        'user': config.pg_user,
        'password': config.pg_password
    }
    
    try:
        migrator = DatabaseMigrator(config.sqlite_path, pg_config)
        validation = migrator.validate_migration()
        
        print("\n" + "=" * 60)
        print("MIGRATION VALIDATION RESULTS")
        print("=" * 60)
        
        all_match = True
        for table, counts in validation.items():
            status = "✓" if counts['match'] else "✗"
            print(f"{status} {table:25} SQLite: {counts['sqlite']:6} | PostgreSQL: {counts['postgresql']:6}")
            
            if not counts['match']:
                all_match = False
        
        print("=" * 60)
        
        if all_match:
            print("✓ All tables validated successfully!")
            return True
        else:
            print("✗ Validation failed - row count mismatches detected")
            return False
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def perform_migration():
    """Perform full migration"""
    logger.info("Starting database migration")
    
    config = DatabaseConfig()
    
    # Confirmation
    print("\n" + "=" * 60)
    print("DATABASE MIGRATION")
    print("=" * 60)
    print(f"Source: SQLite ({config.sqlite_path})")
    print(f"Target: PostgreSQL ({config.pg_user}@{config.pg_host}:{config.pg_port}/{config.pg_database})")
    print("=" * 60)
    
    response = input("\nThis will migrate all data to PostgreSQL. Continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Migration cancelled")
        return False
    
    # Perform migration
    success = migrate_database(
        sqlite_path=config.sqlite_path,
        postgres_host=config.pg_host,
        postgres_port=config.pg_port,
        postgres_db=config.pg_database,
        postgres_user=config.pg_user,
        postgres_password=config.pg_password
    )
    
    if success:
        print("\n✓ Migration completed successfully!")
        
        # Show validation results
        print("\nValidation Results:")
        validate_migration()
        
        return True
    else:
        print("\n✗ Migration failed. Check logs for details.")
        return False


def test_connection():
    """Test database connections"""
    print("\n" + "=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)
    
    config = DatabaseConfig()
    
    # Test SQLite
    print(f"\nTesting SQLite ({config.sqlite_path})...")
    config.db_type = 'sqlite'
    sqlite_conn = DatabaseConnection(config)
    
    if sqlite_conn.health_check():
        print("✓ SQLite connection successful")
        info = sqlite_conn.get_database_info()
        print(f"  Path: {info['path']}")
        print(f"  Exists: {info['exists']}")
    else:
        print("✗ SQLite connection failed")
    
    # Test PostgreSQL
    print(f"\nTesting PostgreSQL ({config.pg_user}@{config.pg_host}:{config.pg_port}/{config.pg_database})...")
    config.db_type = 'postgresql'
    pg_conn = DatabaseConnection(config)
    
    if pg_conn.health_check():
        print("✓ PostgreSQL connection successful")
        info = pg_conn.get_database_info()
        print(f"  Host: {info['host']}")
        print(f"  Database: {info['database']}")
        print(f"  Pool: {info['pool_size']}")
    else:
        print("✗ PostgreSQL connection failed")
    
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate Avalanche database from SQLite to PostgreSQL"
    )
    
    parser.add_argument(
        '--setup-schema',
        action='store_true',
        help='Create PostgreSQL schema and indexes'
    )
    
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Perform full database migration'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing migration'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test database connections'
    )
    
    args = parser.parse_args()
    
    # Show usage if no arguments
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample workflow:")
        print("  1. python scripts/migrate_to_postgres.py --test-connection")
        print("  2. python scripts/migrate_to_postgres.py --setup-schema")
        print("  3. python scripts/migrate_to_postgres.py --migrate")
        print("  4. python scripts/migrate_to_postgres.py --validate")
        return
    
    # Execute requested operation
    if args.test_connection:
        test_connection()
    
    if args.setup_schema:
        if setup_postgres_schema():
            print("\n✓ Schema setup complete")
        else:
            print("\n✗ Schema setup failed")
            sys.exit(1)
    
    if args.migrate:
        if perform_migration():
            print("\n✓ Migration complete")
        else:
            print("\n✗ Migration failed")
            sys.exit(1)
    
    if args.validate:
        if validate_migration():
            print("\n✓ Validation passed")
        else:
            print("\n✗ Validation failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
