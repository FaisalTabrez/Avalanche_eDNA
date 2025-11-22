"""
Database migration utilities for SQLite to PostgreSQL
"""
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class DatabaseMigrator:
    """Migrate data from SQLite to PostgreSQL"""
    
    def __init__(self, sqlite_path: str, postgres_config: Dict[str, Any]):
        """
        Initialize migrator
        
        Args:
            sqlite_path: Path to SQLite database
            postgres_config: PostgreSQL connection parameters
        """
        self.sqlite_path = Path(sqlite_path)
        self.pg_config = postgres_config
        
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("psycopg2 not available. Install psycopg2-binary.")
        
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    
    def _get_sqlite_connection(self):
        """Get SQLite connection"""
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _get_postgres_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.pg_config)
    
    def _convert_sqlite_to_postgres_value(self, value: Any, column_type: str) -> Any:
        """
        Convert SQLite value to PostgreSQL compatible format
        
        Args:
            value: Value from SQLite
            column_type: Expected PostgreSQL column type
            
        Returns:
            Converted value
        """
        if value is None:
            return None
        
        # JSON/JSONB conversion
        if column_type.lower() in ['json', 'jsonb']:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        
        # Timestamp conversion
        if column_type.lower() == 'timestamp':
            if isinstance(value, str):
                # Try parsing ISO format
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return value
            return value
        
        # Array conversion for TEXT[]
        if column_type.endswith('[]'):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return [value]
            return value
        
        return value
    
    def migrate_table(self, table_name: str, column_types: Dict[str, str]) -> int:
        """
        Migrate single table from SQLite to PostgreSQL
        
        Args:
            table_name: Name of table to migrate
            column_types: Mapping of column names to PostgreSQL types
            
        Returns:
            Number of rows migrated
        """
        logger.info(f"Migrating table: {table_name}")
        
        sqlite_conn = self._get_sqlite_connection()
        pg_conn = self._get_postgres_connection()
        
        try:
            # Get data from SQLite
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute(f"SELECT * FROM {table_name}")
            rows = sqlite_cursor.fetchall()
            
            if not rows:
                logger.info(f"No data to migrate for table: {table_name}")
                return 0
            
            # Get column names
            columns = [description[0] for description in sqlite_cursor.description]
            
            # Prepare PostgreSQL insert
            pg_cursor = pg_conn.cursor()
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT DO NOTHING
            """
            
            # Migrate rows
            migrated = 0
            for row in rows:
                # Convert values
                values = []
                for col, val in zip(columns, row):
                    pg_type = column_types.get(col, 'text')
                    converted_val = self._convert_sqlite_to_postgres_value(val, pg_type)
                    values.append(converted_val)
                
                try:
                    pg_cursor.execute(insert_query, values)
                    migrated += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate row in {table_name}: {e}")
                    continue
            
            pg_conn.commit()
            logger.info(f"Migrated {migrated}/{len(rows)} rows for table: {table_name}")
            
            return migrated
        
        finally:
            sqlite_conn.close()
            pg_conn.close()
    
    def migrate_all(self) -> Dict[str, int]:
        """
        Migrate all tables from SQLite to PostgreSQL
        
        Returns:
            Dictionary mapping table names to row counts
        """
        logger.info("Starting full database migration")
        
        # Define column types for each table
        table_schemas = {
            'datasets': {
                'dataset_id': 'varchar',
                'name': 'varchar',
                'description': 'text',
                'sequence_type': 'varchar',
                'file_path': 'text',
                'created_at': 'timestamp',
                'updated_at': 'timestamp',
                'metadata': 'jsonb'
            },
            'analysis_reports': {
                'report_id': 'varchar',
                'dataset_id': 'varchar',
                'analysis_type': 'varchar',
                'status': 'varchar',
                'created_at': 'timestamp',
                'completed_at': 'timestamp',
                'parameters': 'jsonb',
                'results': 'jsonb',
                'error_message': 'text'
            },
            'organism_profiles': {
                'profile_id': 'varchar',
                'report_id': 'varchar',
                'organism_name': 'varchar',
                'taxonomy_id': 'varchar',
                'confidence': 'real',
                'abundance': 'integer',
                'sequence_count': 'integer',
                'metadata': 'jsonb'
            },
            'similarity_matrices': {
                'matrix_id': 'varchar',
                'report_id': 'varchar',
                'matrix_type': 'varchar',
                'data': 'bytea',
                'labels': 'text[]',
                'created_at': 'timestamp'
            },
            'report_comparisons': {
                'comparison_id': 'varchar',
                'report_id_1': 'varchar',
                'report_id_2': 'varchar',
                'comparison_type': 'varchar',
                'similarity_score': 'real',
                'results': 'jsonb',
                'created_at': 'timestamp'
            }
        }
        
        results = {}
        
        # Migrate in order (respecting foreign keys)
        table_order = ['datasets', 'analysis_reports', 'organism_profiles', 
                      'similarity_matrices', 'report_comparisons']
        
        for table in table_order:
            if table in table_schemas:
                try:
                    count = self.migrate_table(table, table_schemas[table])
                    results[table] = count
                except Exception as e:
                    logger.error(f"Failed to migrate table {table}: {e}")
                    results[table] = -1
        
        logger.info("Migration complete")
        return results
    
    def validate_migration(self) -> Dict[str, Dict[str, int]]:
        """
        Validate migration by comparing row counts
        
        Returns:
            Dictionary with row counts from both databases
        """
        logger.info("Validating migration")
        
        sqlite_conn = self._get_sqlite_connection()
        pg_conn = self._get_postgres_connection()
        
        validation = {}
        
        tables = ['datasets', 'analysis_reports', 'organism_profiles',
                 'similarity_matrices', 'report_comparisons']
        
        try:
            for table in tables:
                # SQLite count
                sqlite_cursor = sqlite_conn.cursor()
                sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                sqlite_count = sqlite_cursor.fetchone()[0]
                
                # PostgreSQL count
                pg_cursor = pg_conn.cursor()
                pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                pg_count = pg_cursor.fetchone()[0]
                
                validation[table] = {
                    'sqlite': sqlite_count,
                    'postgresql': pg_count,
                    'match': sqlite_count == pg_count
                }
        
        finally:
            sqlite_conn.close()
            pg_conn.close()
        
        return validation


def migrate_database(
    sqlite_path: str,
    postgres_host: str,
    postgres_port: int,
    postgres_db: str,
    postgres_user: str,
    postgres_password: str
) -> bool:
    """
    Convenience function to migrate entire database
    
    Args:
        sqlite_path: Path to SQLite database
        postgres_host: PostgreSQL host
        postgres_port: PostgreSQL port
        postgres_db: PostgreSQL database name
        postgres_user: PostgreSQL username
        postgres_password: PostgreSQL password
        
    Returns:
        True if migration successful, False otherwise
    """
    pg_config = {
        'host': postgres_host,
        'port': postgres_port,
        'database': postgres_db,
        'user': postgres_user,
        'password': postgres_password
    }
    
    try:
        migrator = DatabaseMigrator(sqlite_path, pg_config)
        
        # Perform migration
        results = migrator.migrate_all()
        
        # Validate
        validation = migrator.validate_migration()
        
        # Check if all tables match
        all_match = all(v['match'] for v in validation.values())
        
        if all_match:
            logger.info("Migration successful and validated")
            return True
        else:
            logger.warning("Migration completed but validation found mismatches")
            for table, counts in validation.items():
                if not counts['match']:
                    logger.warning(f"  {table}: SQLite={counts['sqlite']}, PostgreSQL={counts['postgresql']}")
            return False
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
