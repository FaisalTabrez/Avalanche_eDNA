"""
Authentication Module for Avalanche eDNA System

Provides user authentication, session management, and role-based access control (RBAC).
Supports both SQLite (legacy) and PostgreSQL (production) backends.
"""
import os

from .authenticator import AuthManager, get_auth_manager
from .user_manager import UserManager
from .postgres_user_manager import PostgresUserManager
from .password_utils import (
    PasswordHasher, 
    validate_password_strength, 
    generate_secure_token,
    SessionManager
)
from .decorators import (
    require_auth,
    require_role,
    require_permission,
    require_any_role,
    require_any_permission,
    with_user_context
)


def get_user_manager():
    """
    Factory function to get appropriate user manager based on environment
    
    Returns:
        UserManager (SQLite) or PostgresUserManager (PostgreSQL)
    """
    db_type = os.getenv('DB_TYPE', 'sqlite').lower()
    
    if db_type == 'postgresql':
        return PostgresUserManager()
    else:
        return UserManager()


__all__ = [
    'AuthManager',
    'get_auth_manager',
    'UserManager',
    'PostgresUserManager',
    'get_user_manager',
    'PasswordHasher',
    'validate_password_strength',
    'generate_secure_token',
    'SessionManager',
    'require_auth',
    'require_role',
    'require_permission',
    'require_any_role',
    'require_any_permission',
    'with_user_context'
]
