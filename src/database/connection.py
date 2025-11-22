"""
Database connection factory supporting both SQLite and PostgreSQL
"""
import os
import sqlite3
import logging
from typing import Optional, Any, Union, Dict
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PostgreSQL support
try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not installed. PostgreSQL support disabled.")


class DatabaseConfig:
    """Database configuration from environment variables"""
    
    def __init__(self):
        # Database type selection
        self.db_type = os.getenv('DB_TYPE', 'sqlite').lower()
        
        # SQLite configuration
        self.sqlite_path = os.getenv('SQLITE_PATH', 'data/avalanche.db')
        
        # PostgreSQL configuration
        self.pg_host = os.getenv('DB_HOST', 'localhost')
        self.pg_port = int(os.getenv('DB_PORT', '5432'))
        self.pg_database = os.getenv('DB_NAME', 'avalanche_edna')
        self.pg_user = os.getenv('DB_USER', 'avalanche')
        self.pg_password = os.getenv('DB_PASSWORD', '')
        
        # Connection pool configuration
        self.pool_min_conn = int(os.getenv('DB_POOL_MIN', '2'))
        self.pool_max_conn = int(os.getenv('DB_POOL_MAX', '20'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        
        # Retry configuration
        self.max_retries = int(os.getenv('DB_MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('DB_RETRY_DELAY', '1'))
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL"""
        return self.db_type == 'postgresql' and POSTGRES_AVAILABLE
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite"""
        return self.db_type == 'sqlite'


class ConnectionPool:
    """Connection pool for PostgreSQL"""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize connection pool"""
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("PostgreSQL support not available. Install psycopg2-binary.")
        
        self.config = config
        self._pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Create connection pool"""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.pool_min_conn,
                maxconn=self.config.pool_max_conn,
                host=self.config.pg_host,
                port=self.config.pg_port,
                database=self.config.pg_database,
                user=self.config.pg_user,
                password=self.config.pg_password,
                connect_timeout=self.config.pool_timeout
            )
            logger.info(f"PostgreSQL connection pool initialized: "
                       f"{self.config.pg_user}@{self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database}")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        if self._pool is None:
            raise RuntimeError("Connection pool not initialized")
        return self._pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        if self._pool:
            self._pool.putconn(conn)
    
    def close_all(self):
        """Close all connections in pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed")


class DatabaseConnection:
    """
    Database connection factory supporting SQLite and PostgreSQL
    
    Usage:
        config = DatabaseConfig()
        db_conn = DatabaseConnection(config)
        
        with db_conn.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets")
            results = cursor.fetchall()
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection
        
        Args:
            config: Database configuration (uses defaults if None)
        """
        self.config = config or DatabaseConfig()
        self._pool: Optional[ConnectionPool] = None
        
        # Initialize PostgreSQL pool if needed
        if self.config.is_postgres:
            self._pool = ConnectionPool(self.config)
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection with automatic cleanup
        
        Yields:
            Connection object (sqlite3.Connection or psycopg2.connection)
        """
        if self.config.is_postgres:
            # PostgreSQL connection from pool
            conn = None
            try:
                conn = self._pool.get_connection()
                yield conn
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"PostgreSQL error: {e}")
                raise
            finally:
                if conn:
                    self._pool.return_connection(conn)
        
        else:
            # SQLite connection
            db_path = Path(self.config.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = None
            try:
                conn = sqlite3.connect(str(db_path))
                conn.execute("PRAGMA foreign_keys = ON")
                conn.row_factory = sqlite3.Row
                yield conn
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"SQLite error: {e}")
                raise
            finally:
                if conn:
                    conn.close()
    
    def execute_with_retry(self, query: str, params: tuple = (), max_retries: Optional[int] = None):
        """
        Execute query with retry logic
        
        Args:
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Cursor with query results
        """
        import time
        
        retries = max_retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    return cursor
            except Exception as e:
                last_error = e
                logger.warning(f"Query failed (attempt {attempt + 1}/{retries}): {e}")
                
                if attempt < retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        logger.error(f"Query failed after {retries} attempts")
        raise last_error
    
    def health_check(self) -> bool:
        """
        Check database connectivity
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.config.is_postgres:
                    cursor.execute("SELECT 1")
                else:
                    cursor.execute("SELECT 1")
                
                result = cursor.fetchone()
                return result is not None
        
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information
        
        Returns:
            Dictionary with database metadata
        """
        info = {
            'type': self.config.db_type,
            'healthy': self.health_check()
        }
        
        if self.config.is_postgres:
            info.update({
                'host': self.config.pg_host,
                'port': self.config.pg_port,
                'database': self.config.pg_database,
                'pool_size': f"{self.config.pool_min_conn}-{self.config.pool_max_conn}"
            })
        else:
            info.update({
                'path': self.config.sqlite_path,
                'exists': Path(self.config.sqlite_path).exists()
            })
        
        return info
    
    def close(self):
        """Close database connections"""
        if self._pool:
            self._pool.close_all()


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_database_connection() -> DatabaseConnection:
    """
    Get global database connection instance
    
    Returns:
        DatabaseConnection singleton
    """
    global _db_connection
    
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    
    return _db_connection


def reset_database_connection():
    """Reset global database connection (useful for testing)"""
    global _db_connection
    
    if _db_connection:
        _db_connection.close()
        _db_connection = None
