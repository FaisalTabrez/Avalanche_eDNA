"""
Main authentication manager integrating user management and sessions
"""
import os
import streamlit as st
from typing import Optional, Dict, Tuple
from .password_utils import SessionManager, validate_password_strength
from src.ui import state


class AuthManager:
    """
    Central authentication manager for the application
    """
    
    def __init__(self, user_manager=None, session_timeout: int = 3600):
        """
        Initialize authentication manager
        
        Args:
            user_manager: User manager instance (auto-detected if None)
            session_timeout: Session timeout in seconds (default 1 hour)
        """
        if user_manager is None:
            # Auto-detect based on DB_TYPE environment variable
            db_type = os.getenv('DB_TYPE', 'sqlite').lower()
            if db_type == 'postgresql':
                from .postgres_user_manager import PostgresUserManager
                self.user_manager = PostgresUserManager()
            else:
                from .user_manager import UserManager
                self.user_manager = UserManager()
        else:
            self.user_manager = user_manager
            
        self.session_manager = SessionManager(session_timeout)
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state for authentication"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'session_token' not in st.session_state:
            st.session_state.session_token = None
        if 'admin_created_shown' not in st.session_state:
            st.session_state.admin_created_shown = False
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate user and create session
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (success, message)
        """
        # Authenticate user
        success, user_data = self.user_manager.authenticate(username, password)
        
        if not success:
            return False, "Invalid username or password. Account will be locked after 5 failed attempts."
        
        # Create session
        token = self.session_manager.create_session(
            user_id=user_data['user_id'],
            username=user_data['username'],
            role=user_data['role']
        )
        
        # Update session state
        st.session_state.authenticated = True
        st.session_state.user = user_data
        st.session_state.session_token = token
        
        return True, f"Welcome, {user_data['username']}!"
    
    def logout(self):
        """Logout current user and invalidate session"""
        if st.session_state.session_token:
            self.session_manager.invalidate_session(st.session_state.session_token)
        
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_token = None
    
    def is_authenticated(self) -> bool:
        """
        Check if current user is authenticated
        
        Returns:
            True if authenticated, False otherwise
        """
        # Ensure session state is initialized
        self._init_session_state()
        
        if not st.session_state.authenticated:
            return False
        
        # Validate session
        token = st.session_state.session_token
        if not token:
            return False
        
        is_valid, session_data = self.session_manager.validate_session(token)
        
        if not is_valid:
            # Session expired - logout
            self.logout()
            return False
        
        return True
    
    def get_current_user(self) -> Optional[Dict]:
        """
        Get current authenticated user
        
        Returns:
            User data or None
        """
        if not self.is_authenticated():
            return None
        
        return st.session_state.user
    
    def has_role(self, role: str) -> bool:
        """
        Check if current user has a specific role
        
        Args:
            role: Role to check
            
        Returns:
            True if user has role, False otherwise
        """
        user = self.get_current_user()
        if not user:
            return False
        
        return user['role'] == role
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if current user has a specific permission
        
        Args:
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        user = self.get_current_user()
        if not user:
            return False
        
        return self.user_manager.has_permission(user['user_id'], permission)
    
    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        confirm_password: str,
        role: str = 'viewer'
    ) -> Tuple[bool, str]:
        """
        Register a new user
        
        Args:
            username: Username
            email: Email
            password: Password
            confirm_password: Password confirmation
            role: User role (default: viewer)
            
        Returns:
            Tuple of (success, message)
        """
        # Validate passwords match
        if password != confirm_password:
            return False, "Passwords do not match"
        
        # Validate password strength
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            return False, error_msg
        
        # Create user
        success, result = self.user_manager.create_user(username, email, password, role)
        
        if not success:
            return False, result
        
        return True, "User registered successfully. You can now log in."
    
    def change_password(
        self,
        current_password: str,
        new_password: str,
        confirm_password: str
    ) -> Tuple[bool, str]:
        """
        Change current user's password
        
        Args:
            current_password: Current password
            new_password: New password
            confirm_password: New password confirmation
            
        Returns:
            Tuple of (success, message)
        """
        user = self.get_current_user()
        if not user:
            return False, "Not authenticated"
        
        # Verify current password
        success, _ = self.user_manager.authenticate(user['username'], current_password)
        if not success:
            return False, "Current password is incorrect"
        
        # Validate new password
        if new_password != confirm_password:
            return False, "New passwords do not match"
        
        is_valid, error_msg = validate_password_strength(new_password)
        if not is_valid:
            return False, error_msg
        
        # Change password
        return self.user_manager.change_password(user['user_id'], new_password)
    
    def require_auth(self, redirect_to: str = "login"):
        """
        Require authentication for a page
        
        Args:
            redirect_to: Page to redirect to if not authenticated
        """
        if not self.is_authenticated():
            st.warning("Please log in to access this page.")
            state.set('current_page_key', redirect_to)
            st.rerun()
    
    def require_role(self, role: str, redirect_to: str = "home"):
        """
        Require specific role for a page
        
        Args:
            role: Required role
            redirect_to: Page to redirect to if role not matched
        """
        self.require_auth()
        
        if not self.has_role(role):
            st.error(f"Access denied. This page requires {role} role.")
            state.set('current_page_key', redirect_to)
            st.rerun()
    
    def require_permission(self, permission: str, redirect_to: str = "home"):
        """
        Require specific permission for a page
        
        Args:
            permission: Required permission
            redirect_to: Page to redirect to if permission not granted
        """
        self.require_auth()
        
        if not self.has_permission(permission):
            st.error(f"Access denied. This action requires '{permission}' permission.")
            state.set('current_page_key', redirect_to)
            st.rerun()
    
    def create_default_admin(self, username: str = "admin", password: str = "Admin@123"):
        """
        Create default admin user if no users exist
        
        Args:
            username: Admin username
            password: Admin password
            
        Returns:
            Tuple of (created, message)
        """
        users = self.user_manager.list_users()
        
        if len(users) > 0:
            return False, "Users already exist"
        
        success, result = self.user_manager.create_user(
            username=username,
            email="admin@avalanche.local",
            password=password,
            role="admin"
        )
        
        if success:
            return True, f"Default admin user created. Username: {username}, Password: {password}"
        
        return False, result


# Global authentication manager instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """
    Get the global authentication manager instance
    
    Returns:
        AuthManager instance
    """
    global _auth_manager
    
    if _auth_manager is None:
        _auth_manager = AuthManager()
    
    return _auth_manager
