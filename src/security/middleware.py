"""
Security middleware and decorators for Streamlit application
"""
import streamlit as st
import functools
import logging
from typing import Callable, Optional, Dict, Any
from pathlib import Path

from .validators import get_rate_limiter, InputSanitizer

logger = logging.getLogger(__name__)


def rate_limit(max_requests: int = 60, window_seconds: int = 60):
    """
    Decorator to rate limit function calls
    
    Args:
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
        
    Usage:
        @rate_limit(max_requests=10, window_seconds=60)
        def expensive_operation():
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            
            # Use session ID as identifier
            identifier = st.session_state.get('session_id', 'anonymous')
            
            allowed, retry_after = limiter.is_allowed(
                identifier,
                max_requests,
                window_seconds
            )
            
            if not allowed:
                st.error(f"⚠️ Rate limit exceeded. Please try again in {retry_after} seconds.")
                st.stop()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_file_upload(
    check_mime: bool = True,
    check_malicious: bool = True
):
    """
    Decorator to validate file uploads
    
    Args:
        check_mime: Whether to check MIME type
        check_malicious: Whether to scan for malicious content
        
    Usage:
        @validate_file_upload()
        def process_upload(uploaded_file):
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(uploaded_file, *args, **kwargs):
            from .validators import FileValidator
            
            if uploaded_file is None:
                st.error("No file uploaded")
                st.stop()
            
            # Validate filename
            valid, error = FileValidator.validate_filename(uploaded_file.name)
            if not valid:
                st.error(f"❌ Invalid filename: {error}")
                st.stop()
            
            # Validate file size
            file_size = uploaded_file.size
            valid, error = FileValidator.validate_file_size(file_size)
            if not valid:
                st.error(f"❌ {error}")
                st.stop()
            
            # Save temporarily for content validation
            temp_path = Path(f"/tmp/{FileValidator.sanitize_filename(uploaded_file.name)}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Validate file content
                valid, errors = FileValidator.validate_file(
                    temp_path,
                    uploaded_file.name,
                    check_mime=check_mime,
                    check_malicious=check_malicious
                )
                
                if not valid:
                    for error in errors:
                        st.error(f"❌ {error}")
                    st.stop()
                
                # Pass validated file to function
                return func(uploaded_file, *args, **kwargs)
            
            finally:
                # Cleanup temp file
                if temp_path.exists():
                    temp_path.unlink()
        
        return wrapper
    return decorator


def sanitize_input(param_name: str, max_length: int = 1000):
    """
    Decorator to sanitize text input parameters
    
    Args:
        param_name: Name of parameter to sanitize
        max_length: Maximum allowed length
        
    Usage:
        @sanitize_input('dataset_name', max_length=100)
        def create_dataset(dataset_name: str):
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize if parameter exists
            if param_name in kwargs:
                original = kwargs[param_name]
                sanitized = InputSanitizer.sanitize_text(original, max_length)
                
                if sanitized != original:
                    logger.warning(f"Input sanitized for {param_name}: {original[:50]}... -> {sanitized[:50]}...")
                
                kwargs[param_name] = sanitized
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class SecurityHeaders:
    """
    Add security headers to responses (for FastAPI integration)
    """
    
    HEADERS = {
        # Prevent clickjacking
        'X-Frame-Options': 'DENY',
        
        # Prevent MIME type sniffing
        'X-Content-Type-Options': 'nosniff',
        
        # Enable XSS protection
        'X-XSS-Protection': '1; mode=block',
        
        # Referrer policy
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        
        # Content Security Policy
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:;"
        ),
        
        # Permissions policy
        'Permissions-Policy': (
            'geolocation=(), '
            'microphone=(), '
            'camera=(), '
            'payment=(), '
            'usb=()'
        ),
    }
    
    @classmethod
    def add_to_response(cls, response):
        """
        Add security headers to response object
        
        Args:
            response: Response object (FastAPI or similar)
        """
        for header, value in cls.HEADERS.items():
            response.headers[header] = value
        
        return response


def initialize_session_security():
    """
    Initialize security features in Streamlit session
    """
    # Generate session ID if not exists
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    # Track session creation time
    if 'session_created_at' not in st.session_state:
        from datetime import datetime
        st.session_state.session_created_at = datetime.now()
    
    # Initialize request counter
    if 'request_count' not in st.session_state:
        st.session_state.request_count = 0
    
    st.session_state.request_count += 1


def check_path_traversal(path: str, base_dir: str) -> bool:
    """
    Check if path attempts directory traversal outside base directory
    
    Args:
        path: Path to check
        base_dir: Base directory that should contain the path
        
    Returns:
        True if path is safe, False if traversal detected
    """
    try:
        # Resolve both paths
        resolved_path = Path(path).resolve()
        resolved_base = Path(base_dir).resolve()
        
        # Check if resolved path is within base directory
        return resolved_path.is_relative_to(resolved_base)
    
    except Exception as e:
        logger.warning(f"Path traversal check failed: {e}")
        return False


def validate_dataset_path(dataset_name: str, base_dir: str = "data") -> Path:
    """
    Validate and construct safe dataset path
    
    Args:
        dataset_name: User-provided dataset name
        base_dir: Base directory for datasets
        
    Returns:
        Safe Path object
        
    Raises:
        ValueError: If path validation fails
    """
    # Sanitize dataset name
    safe_name = InputSanitizer.sanitize_dataset_name(dataset_name)
    
    # Construct path
    dataset_path = Path(base_dir) / safe_name
    
    # Verify no traversal
    if not check_path_traversal(dataset_path, base_dir):
        raise ValueError(f"Invalid dataset path: {dataset_name}")
    
    return dataset_path


def log_security_event(event_type: str, details: Dict[str, Any]):
    """
    Log security-relevant events
    
    Args:
        event_type: Type of security event
        details: Event details
    """
    from datetime import datetime
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'session_id': st.session_state.get('session_id', 'unknown'),
        'user': st.session_state.get('user', {}).get('username', 'anonymous'),
        'details': details
    }
    
    logger.warning(f"SECURITY EVENT: {log_entry}")
