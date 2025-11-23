"""
PostgreSQL-based User Manager with Redis caching
Streamlined version for better performance and concurrency
"""
import os
import uuid
import json
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import redis

logger = logging.getLogger(__name__)

try:
    from .password_utils import PasswordHasher
except ImportError:
    from src.auth.password_utils import PasswordHasher


class PostgresUserManager:
    """
    User manager using PostgreSQL and Redis for optimal performance
    """
    
    # Supported user roles with permissions
    ROLES = {
        'admin': {
            'description': 'Full system access',
            'permissions': ['read', 'write', 'delete', 'manage_users', 'manage_system']
        },
        'analyst': {
            'description': 'Can create and analyze datasets',
            'permissions': ['read', 'write', 'delete_own']
        },
        'viewer': {
            'description': 'Read-only access',
            'permissions': ['read']
        }
    }
    
    def __init__(
        self,
        pg_host: str = None,
        pg_port: int = None,
        pg_database: str = None,
        pg_user: str = None,
        pg_password: str = None,
        redis_url: str = None
    ):
        """
        Initialize user manager with PostgreSQL and Redis
        
        Args:
            pg_host: PostgreSQL host (defaults to env DB_HOST)
            pg_port: PostgreSQL port (defaults to env DB_PORT)
            pg_database: Database name (defaults to env DB_NAME)
            pg_user: Database user (defaults to env DB_USER)
            pg_password: Database password (defaults to env DB_PASSWORD)
            redis_url: Redis connection URL (defaults to env REDIS_URL)
        """
        # PostgreSQL configuration
        self.pg_host = pg_host or os.getenv('DB_HOST', 'localhost')
        self.pg_port = pg_port or int(os.getenv('DB_PORT', '5432'))
        self.pg_database = pg_database or os.getenv('DB_NAME', 'avalanche_edna')
        self.pg_user = pg_user or os.getenv('DB_USER', 'avalanche')
        self.pg_password = pg_password or os.getenv('DB_PASSWORD', 'avalanche_dev_password')
        
        # Redis configuration
        redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            self.redis_client = None
            self.redis_available = False
        
        # Initialize connection pool
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=self.pg_host,
                port=self.pg_port,
                database=self.pg_database,
                user=self.pg_user,
                password=self.pg_password
            )
            logger.info(f"PostgreSQL connection pool initialized: {self.pg_user}@{self.pg_host}:{self.pg_port}/{self.pg_database}")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise
        
        self._init_database()
    
    def _get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def _return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def _init_database(self):
        """Initialize the users database schema"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id UUID PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ip_address VARCHAR(50),
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            """)
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    user_id UUID,
                    action VARCHAR(100) NOT NULL,
                    resource VARCHAR(255),
                    details TEXT,
                    ip_address VARCHAR(50),
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
        finally:
            self._return_connection(conn)
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = 'viewer'
    ) -> Tuple[bool, str]:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password (will be hashed)
            role: User role (admin, analyst, viewer)
            
        Returns:
            Tuple of (success, user_id or error message)
        """
        # Validate role
        if role not in self.ROLES:
            return False, f"Invalid role. Must be one of: {', '.join(self.ROLES.keys())}"
        
        # Hash password
        password_hash = PasswordHasher.hash_password(password)
        
        # Generate user ID
        user_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users (
                    user_id, username, email, password_hash, role, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                username,
                email,
                password_hash,
                role,
                datetime.now()
            ))
            
            conn.commit()
            
            self._log_action(user_id, 'user_created', f'username={username}')
            self._invalidate_cache('users_list')
            
            return True, user_id
            
        except psycopg2.IntegrityError as e:
            conn.rollback()
            error_msg = str(e).lower()
            if 'username' in error_msg:
                return False, "Username already exists"
            elif 'email' in error_msg:
                return False, "Email already exists"
            return False, str(e)
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating user: {e}")
            return False, str(e)
        finally:
            self._return_connection(conn)
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """
        Authenticate a user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (success, user_data or None)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get user
            cursor.execute("""
                SELECT * FROM users WHERE username = %s
            """, (username,))
            
            row = cursor.fetchone()
            
            if not row:
                return False, None
            
            user = dict(row)
            
            # Check if account is locked
            if user['locked_until']:
                if datetime.now() < user['locked_until']:
                    return False, None
            
            # Check if account is active
            if not user['is_active']:
                return False, None
            
            # Verify password
            if not PasswordHasher.verify_password(password, user['password_hash']):
                # Increment failed login attempts
                cursor.execute("""
                    UPDATE users 
                    SET failed_login_attempts = failed_login_attempts + 1
                    WHERE username = %s
                """, (username,))
                
                # Lock account after 5 failed attempts for 15 minutes
                if user['failed_login_attempts'] + 1 >= 5:
                    locked_until = datetime.now() + timedelta(minutes=15)
                    cursor.execute("""
                        UPDATE users 
                        SET locked_until = %s
                        WHERE username = %s
                    """, (locked_until, username))
                
                conn.commit()
                return False, None
            
            # Successful authentication - reset failed attempts
            cursor.execute("""
                UPDATE users 
                SET failed_login_attempts = 0,
                    locked_until = NULL,
                    last_login = %s
                WHERE username = %s
            """, (datetime.now(), username))
            
            conn.commit()
            
            # Remove sensitive data before returning
            user_data = {
                'user_id': str(user['user_id']),
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
            
            self._log_action(str(user['user_id']), 'login_success', f'username={username}')
            
            # Cache user data
            self._cache_user(user_data)
            
            return True, user_data
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Authentication error: {e}")
            return False, None
        finally:
            self._return_connection(conn)
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User data or None
        """
        # Try cache first
        cached = self._get_cached_user(user_id)
        if cached:
            return cached
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT user_id, username, email, role, created_at, last_login, is_active
                FROM users WHERE user_id = %s
            """, (user_id,))
            
            row = cursor.fetchone()
            
            if row:
                user_data = dict(row)
                user_data['user_id'] = str(user_data['user_id'])
                user_data['created_at'] = user_data['created_at'].isoformat() if user_data['created_at'] else None
                user_data['last_login'] = user_data['last_login'].isoformat() if user_data['last_login'] else None
                
                # Cache result
                self._cache_user(user_data)
                
                return user_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
        finally:
            self._return_connection(conn)
    
    def list_users(self) -> List[Dict]:
        """
        List all users
        
        Returns:
            List of user data dictionaries
        """
        # Try cache first
        cached = self._get_cached('users_list')
        if cached:
            return json.loads(cached)
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT user_id, username, email, role, created_at, last_login, is_active
                FROM users
                ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            
            users = []
            for row in rows:
                user = dict(row)
                user['user_id'] = str(user['user_id'])
                user['created_at'] = user['created_at'].isoformat() if user['created_at'] else None
                user['last_login'] = user['last_login'].isoformat() if user['last_login'] else None
                users.append(user)
            
            # Cache result for 60 seconds
            self._cache_set('users_list', json.dumps(users), ttl=60)
            
            return users
            
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Tuple[bool, str]:
        """
        Update user information
        
        Args:
            user_id: User ID
            email: New email (optional)
            role: New role (optional)
            is_active: Active status (optional)
            
        Returns:
            Tuple of (success, message)
        """
        updates = []
        params = []
        
        if email is not None:
            updates.append("email = %s")
            params.append(email)
        
        if role is not None:
            if role not in self.ROLES:
                return False, f"Invalid role. Must be one of: {', '.join(self.ROLES.keys())}"
            updates.append("role = %s")
            params.append(role)
        
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(is_active)
        
        if not updates:
            return False, "No updates provided"
        
        params.append(user_id)
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute(f"""
                UPDATE users 
                SET {', '.join(updates)}
                WHERE user_id = %s
            """, params)
            
            if cursor.rowcount == 0:
                conn.rollback()
                return False, "User not found"
            
            conn.commit()
            
            self._log_action(user_id, 'user_updated', f'fields={",".join(updates)}')
            self._invalidate_cache(f'user:{user_id}')
            self._invalidate_cache('users_list')
            
            return True, "User updated successfully"
            
        except psycopg2.IntegrityError as e:
            conn.rollback()
            return False, str(e)
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating user: {e}")
            return False, str(e)
        finally:
            self._return_connection(conn)
    
    def change_password(self, user_id: str, new_password: str) -> Tuple[bool, str]:
        """
        Change user password
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        password_hash = PasswordHasher.hash_password(new_password)
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE users 
                SET password_hash = %s
                WHERE user_id = %s
            """, (password_hash, user_id))
            
            if cursor.rowcount == 0:
                conn.rollback()
                return False, "User not found"
            
            conn.commit()
            
            self._log_action(user_id, 'password_changed', '')
            
            return True, "Password changed successfully"
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error changing password: {e}")
            return False, str(e)
        finally:
            self._return_connection(conn)
    
    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Delete user (cascade will handle sessions)
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            
            if cursor.rowcount == 0:
                conn.rollback()
                return False, "User not found"
            
            conn.commit()
            
            self._log_action(user_id, 'user_deleted', '')
            self._invalidate_cache(f'user:{user_id}')
            self._invalidate_cache('users_list')
            
            return True, "User deleted successfully"
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting user: {e}")
            return False, str(e)
        finally:
            self._return_connection(conn)
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        user = self.get_user(user_id)
        if not user:
            return False
        
        role = user['role']
        if role not in self.ROLES:
            return False
        
        return permission in self.ROLES[role]['permissions']
    
    def _log_action(self, user_id: Optional[str], action: str, details: str, ip_address: Optional[str] = None):
        """
        Log an action to the audit log
        
        Args:
            user_id: User ID (can be None for system actions)
            action: Action performed
            details: Additional details
            ip_address: IP address of the request
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Only log user_id if user still exists (avoid FK constraint violations)
            if user_id:
                cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
                if not cursor.fetchone():
                    user_id = None  # User was deleted, don't reference them
            
            cursor.execute("""
                INSERT INTO audit_log (timestamp, user_id, action, details, ip_address)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                datetime.now(),
                user_id if user_id else None,
                action,
                details,
                ip_address
            ))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error logging action: {e}")
        finally:
            self._return_connection(conn)
    
    # Redis caching methods
    def _cache_user(self, user_data: Dict, ttl: int = 300):
        """Cache user data"""
        if self.redis_available and user_data:
            try:
                key = f"user:{user_data['user_id']}"
                self.redis_client.setex(key, ttl, json.dumps(user_data))
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
    
    def _get_cached_user(self, user_id: str) -> Optional[Dict]:
        """Get cached user data"""
        if self.redis_available:
            try:
                key = f"user:{user_id}"
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        return None
    
    def _cache_set(self, key: str, value: str, ttl: int = 300):
        """Set cache value"""
        if self.redis_available:
            try:
                self.redis_client.setex(key, ttl, value)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
    
    def _get_cached(self, key: str) -> Optional[str]:
        """Get cached value"""
        if self.redis_available:
            try:
                return self.redis_client.get(key)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        return None
    
    def _invalidate_cache(self, key: str):
        """Invalidate cache key"""
        if self.redis_available:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")
    
    def close(self):
        """Close all connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
        if self.redis_client:
            self.redis_client.close()
