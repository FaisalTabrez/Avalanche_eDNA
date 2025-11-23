"""
Decorators for authentication and authorization
"""
from functools import wraps
import streamlit as st
from typing import Callable, Optional
from src.ui import state


def require_auth(redirect_to: str = "login"):
    """
    Decorator to require authentication for a function
    
    Args:
        redirect_to: Page to redirect to if not authenticated
        
    Usage:
        @require_auth()
        def protected_function():
            # This code only runs if user is authenticated
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .authenticator import get_auth_manager
            
            auth = get_auth_manager()
            
            if not auth.is_authenticated():
                st.warning("Please log in to access this feature.")
                state.set('current_page_key', redirect_to)
                st.rerun()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: str, redirect_to: str = "home"):
    """
    Decorator to require specific role for a function
    
    Args:
        role: Required role (admin, analyst, viewer)
        redirect_to: Page to redirect to if role not matched
        
    Usage:
        @require_role('admin')
        def admin_only_function():
            # This code only runs if user is admin
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .authenticator import get_auth_manager
            
            auth = get_auth_manager()
            
            if not auth.is_authenticated():
                st.warning("Please log in to access this feature.")
                state.set('current_page_key', 'login')
                st.rerun()
            
            if not auth.has_role(role):
                st.error(f"Access denied. This feature requires {role} role.")
                state.set('current_page_key', redirect_to)
                st.rerun()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_permission(permission: str, redirect_to: str = "home"):
    """
    Decorator to require specific permission for a function
    
    Args:
        permission: Required permission (read, write, delete, manage_users, etc.)
        redirect_to: Page to redirect to if permission not granted
        
    Usage:
        @require_permission('manage_users')
        def manage_users_function():
            # This code only runs if user has manage_users permission
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .authenticator import get_auth_manager
            
            auth = get_auth_manager()
            
            if not auth.is_authenticated():
                st.warning("Please log in to access this feature.")
                state.set('current_page_key', 'login')
                st.rerun()
            
            if not auth.has_permission(permission):
                st.error(f"Access denied. This action requires '{permission}' permission.")
                state.set('current_page_key', redirect_to)
                st.rerun()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_any_role(*roles: str, redirect_to: str = "home"):
    """
    Decorator to require any of the specified roles
    
    Args:
        *roles: Required roles (user must have at least one)
        redirect_to: Page to redirect to if no roles match
        
    Usage:
        @require_any_role('admin', 'analyst')
        def admin_or_analyst_function():
            # This code runs if user is admin OR analyst
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .authenticator import get_auth_manager
            
            auth = get_auth_manager()
            
            if not auth.is_authenticated():
                st.warning("Please log in to access this feature.")
                state.set('current_page_key', 'login')
                st.rerun()
            
            user = auth.get_current_user()
            if user['role'] not in roles:
                st.error(f"Access denied. This feature requires one of: {', '.join(roles)}")
                state.set('current_page_key', redirect_to)
                st.rerun()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_any_permission(*permissions: str, redirect_to: str = "home"):
    """
    Decorator to require any of the specified permissions
    
    Args:
        *permissions: Required permissions (user must have at least one)
        redirect_to: Page to redirect to if no permissions match
        
    Usage:
        @require_any_permission('write', 'delete')
        def modify_data_function():
            # This code runs if user has write OR delete permission
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .authenticator import get_auth_manager
            
            auth = get_auth_manager()
            
            if not auth.is_authenticated():
                st.warning("Please log in to access this feature.")
                state.set('current_page_key', 'login')
                st.rerun()
            
            has_permission = any(auth.has_permission(perm) for perm in permissions)
            
            if not has_permission:
                st.error(f"Access denied. This action requires one of: {', '.join(permissions)}")
                state.set('current_page_key', redirect_to)
                st.rerun()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_user_context(func: Callable):
    """
    Decorator that injects current user as first argument to function
    
    Usage:
        @with_user_context
        def my_function(user, other_arg):
            print(f"User: {user['username']}")
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from .authenticator import get_auth_manager
        
        auth = get_auth_manager()
        user = auth.get_current_user()
        
        return func(user, *args, **kwargs)
    
    return wrapper
