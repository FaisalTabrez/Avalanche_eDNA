"""
Authentication Module for Avalanche eDNA System

Provides user authentication, session management, and role-based access control (RBAC).
"""

from .authenticator import AuthManager, get_auth_manager
from .user_manager import UserManager
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

__all__ = [
    'AuthManager',
    'get_auth_manager',
    'UserManager',
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

__all__ = [
    'AuthManager',
    'UserManager',
    'PasswordHasher',
    'validate_password_strength',
    'require_auth',
    'require_role',
]
