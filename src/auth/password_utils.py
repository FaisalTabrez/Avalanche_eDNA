"""
Password utilities for secure password hashing and validation
"""
import hashlib
import secrets
import re
from typing import Tuple, Dict
from datetime import datetime, timedelta


class PasswordHasher:
    """
    Secure password hashing using PBKDF2-SHA256
    """
    
    ITERATIONS = 100000  # OWASP recommended minimum
    HASH_ALGORITHM = 'sha256'
    SALT_LENGTH = 32
    
    @classmethod
    def hash_password(cls, password: str) -> str:
        """
        Hash a password with a random salt
        
        Args:
            password: Plain text password
            
        Returns:
            String in format: algorithm$iterations$salt$hash
        """
        salt = secrets.token_hex(cls.SALT_LENGTH)
        pwd_hash = hashlib.pbkdf2_hmac(
            cls.HASH_ALGORITHM,
            password.encode('utf-8'),
            salt.encode('utf-8'),
            cls.ITERATIONS
        )
        hash_hex = pwd_hash.hex()
        
        return f"{cls.HASH_ALGORITHM}${cls.ITERATIONS}${salt}${hash_hex}"
    
    @classmethod
    def verify_password(cls, password: str, hashed: str) -> bool:
        """
        Verify a password against a hash
        
        Args:
            password: Plain text password to verify
            hashed: Stored hash in format algorithm$iterations$salt$hash
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            parts = hashed.split('$')
            if len(parts) != 4:
                return False
            
            algorithm, iterations, salt, stored_hash = parts
            iterations = int(iterations)
            
            # Hash the provided password with the same salt
            pwd_hash = hashlib.pbkdf2_hmac(
                algorithm,
                password.encode('utf-8'),
                salt.encode('utf-8'),
                iterations
            )
            
            # Constant-time comparison to prevent timing attacks
            return secrets.compare_digest(pwd_hash.hex(), stored_hash)
            
        except (ValueError, AttributeError):
            return False


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password strength against security requirements
    
    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if len(password) > 128:
        return False, "Password must be less than 128 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    
    # Check for common weak passwords
    weak_passwords = ['password', '12345678', 'qwerty', 'abc123', 'password123']
    if password.lower() in weak_passwords:
        return False, "This password is too common. Please choose a stronger password"
    
    return True, ""


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token
    
    Args:
        length: Number of bytes for the token
        
    Returns:
        Hex string token
    """
    return secrets.token_hex(length)


class SessionManager:
    """
    Manage user sessions with automatic expiration
    """
    
    def __init__(self, timeout_seconds: int = 3600):
        """
        Initialize session manager
        
        Args:
            timeout_seconds: Session timeout in seconds (default 1 hour)
        """
        self.timeout = timedelta(seconds=timeout_seconds)
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, user_id: str, username: str, role: str) -> str:
        """
        Create a new session for a user
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            
        Returns:
            Session token
        """
        token = generate_secure_token()
        
        self.sessions[token] = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        return token
    
    def validate_session(self, token: str) -> Tuple[bool, Dict]:
        """
        Validate a session token
        
        Args:
            token: Session token
            
        Returns:
            Tuple of (is_valid, session_data)
        """
        if token not in self.sessions:
            return False, {}
        
        session = self.sessions[token]
        
        # Check if session has expired
        if datetime.now() - session['last_activity'] > self.timeout:
            del self.sessions[token]
            return False, {}
        
        # Update last activity
        session['last_activity'] = datetime.now()
        
        return True, session
    
    def invalidate_session(self, token: str):
        """
        Invalidate/delete a session
        
        Args:
            token: Session token to invalidate
        """
        if token in self.sessions:
            del self.sessions[token]
    
    def cleanup_expired_sessions(self):
        """
        Remove all expired sessions
        """
        now = datetime.now()
        expired = [
            token for token, session in self.sessions.items()
            if now - session['last_activity'] > self.timeout
        ]
        
        for token in expired:
            del self.sessions[token]
