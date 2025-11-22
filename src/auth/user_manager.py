"""
User management for authentication system
"""
import sqlite3
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path
from .password_utils import PasswordHasher


class UserManager:
    """
    Manage users in the authentication system
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
    
    def __init__(self, db_path: str = "data/avalanche_users.db"):
        """
        Initialize user manager
        
        Args:
            db_path: Path to SQLite database for users
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the users database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TEXT
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                details TEXT,
                ip_address TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
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
        import uuid
        user_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users (
                    user_id, username, email, password_hash, role, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                username,
                email,
                password_hash,
                role,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self._log_action(user_id, 'user_created', f'username={username}')
            
            return True, user_id
            
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return False, "Username already exists"
            elif 'email' in str(e):
                return False, "Email already exists"
            return False, str(e)
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """
        Authenticate a user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (success, user_data or None)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user
        cursor.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False, None
        
        user = dict(row)
        
        # Check if account is locked
        if user['locked_until']:
            locked_until = datetime.fromisoformat(user['locked_until'])
            if datetime.now() < locked_until:
                conn.close()
                return False, None
        
        # Check if account is active
        if not user['is_active']:
            conn.close()
            return False, None
        
        # Verify password
        if not PasswordHasher.verify_password(password, user['password_hash']):
            # Increment failed login attempts
            cursor.execute("""
                UPDATE users 
                SET failed_login_attempts = failed_login_attempts + 1
                WHERE username = ?
            """, (username,))
            
            # Lock account after 5 failed attempts for 15 minutes
            if user['failed_login_attempts'] + 1 >= 5:
                from datetime import timedelta
                locked_until = datetime.now() + timedelta(minutes=15)
                cursor.execute("""
                    UPDATE users 
                    SET locked_until = ?
                    WHERE username = ?
                """, (locked_until.isoformat(), username))
            
            conn.commit()
            conn.close()
            return False, None
        
        # Successful authentication - reset failed attempts
        cursor.execute("""
            UPDATE users 
            SET failed_login_attempts = 0,
                locked_until = NULL,
                last_login = ?
            WHERE username = ?
        """, (datetime.now().isoformat(), username))
        
        conn.commit()
        conn.close()
        
        # Remove sensitive data before returning
        user_data = {
            'user_id': user['user_id'],
            'username': user['username'],
            'email': user['email'],
            'role': user['role']
        }
        
        self._log_action(user['user_id'], 'login_success', f'username={username}')
        
        return True, user_data
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User data or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, role, created_at, last_login, is_active
            FROM users WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def list_users(self) -> List[Dict]:
        """
        List all users
        
        Returns:
            List of user data dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, role, created_at, last_login, is_active
            FROM users
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
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
            updates.append("email = ?")
            params.append(email)
        
        if role is not None:
            if role not in self.ROLES:
                return False, f"Invalid role. Must be one of: {', '.join(self.ROLES.keys())}"
            updates.append("role = ?")
            params.append(role)
        
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        
        if not updates:
            return False, "No updates provided"
        
        params.append(user_id)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                UPDATE users 
                SET {', '.join(updates)}
                WHERE user_id = ?
            """, params)
            
            if cursor.rowcount == 0:
                conn.close()
                return False, "User not found"
            
            conn.commit()
            conn.close()
            
            self._log_action(user_id, 'user_updated', f'fields={",".join(updates)}')
            
            return True, "User updated successfully"
            
        except sqlite3.IntegrityError as e:
            return False, str(e)
    
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
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET password_hash = ?
            WHERE user_id = ?
        """, (password_hash, user_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return False, "User not found"
        
        conn.commit()
        conn.close()
        
        self._log_action(user_id, 'password_changed', '')
        
        return True, "Password changed successfully"
    
    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, message)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete user's sessions first
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            return False, "User not found"
        
        conn.commit()
        conn.close()
        
        self._log_action(user_id, 'user_deleted', '')
        
        return True, "User deleted successfully"
    
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_log (timestamp, user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_id,
            action,
            details,
            ip_address
        ))
        
        conn.commit()
        conn.close()
