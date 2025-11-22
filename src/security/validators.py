"""
Input validation and security utilities for file uploads and user inputs
"""
import os
import re
import hashlib
import magic
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """
    Validate uploaded files for security and format compliance
    """
    
    # Allowed file extensions for bioinformatics data
    ALLOWED_EXTENSIONS = {
        '.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn',  # FASTA formats
        '.fastq', '.fq',  # FASTQ formats
        '.gz', '.bz2', '.zip',  # Compressed formats
        '.csv', '.tsv', '.txt',  # Tabular data
        '.json', '.yaml', '.yml',  # Configuration files
        '.gbk', '.gb', '.genbank',  # GenBank format
        '.sam', '.bam',  # Alignment formats
        '.vcf',  # Variant call format
    }
    
    # MIME types for allowed files
    ALLOWED_MIME_TYPES = {
        'text/plain',
        'text/csv',
        'application/json',
        'application/gzip',
        'application/x-gzip',
        'application/zip',
        'application/x-zip-compressed',
        'application/x-bzip2',
        'application/octet-stream',  # Generic binary (for .bam, .gz, etc.)
    }
    
    # Maximum file size (configurable via environment)
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '500'))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Dangerous patterns in filenames
    DANGEROUS_PATTERNS = [
        r'\.\.',  # Directory traversal
        r'[<>:"|?*]',  # Invalid filename characters
        r'^[\/\\]',  # Absolute paths
        r'[\x00-\x1f]',  # Control characters
    ]
    
    @classmethod
    def validate_filename(cls, filename: str) -> Tuple[bool, str]:
        """
        Validate filename for security issues
        
        Args:
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename is empty"
        
        # Check length
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                return False, f"Filename contains invalid characters: {pattern}"
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"File extension '{file_ext}' not allowed. Allowed: {', '.join(sorted(cls.ALLOWED_EXTENSIONS))}"
        
        return True, ""
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename by removing dangerous characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Get base name (no directory components)
        filename = os.path.basename(filename)
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f]', '', filename)
        
        # Replace dangerous characters with underscore
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        # Collapse multiple dots (except for extensions)
        parts = filename.split('.')
        if len(parts) > 2:
            # Keep only last extension
            filename = '_'.join(parts[:-1]) + '.' + parts[-1]
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not empty after sanitization
        if not filename:
            filename = "unnamed_file"
        
        return filename
    
    @classmethod
    def validate_file_size(cls, file_size: int) -> Tuple[bool, str]:
        """
        Validate file size
        
        Args:
            file_size: File size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if file_size <= 0:
            return False, "File is empty"
        
        if file_size > cls.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum allowed: {cls.MAX_FILE_SIZE_MB}MB"
        
        return True, ""
    
    @classmethod
    def validate_mime_type(cls, file_path: Path) -> Tuple[bool, str]:
        """
        Validate file MIME type using python-magic
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))
            
            if mime_type not in cls.ALLOWED_MIME_TYPES:
                return False, f"File type '{mime_type}' not allowed"
            
            return True, ""
        
        except Exception as e:
            logger.warning(f"Failed to detect MIME type: {e}")
            # Allow if MIME detection fails (but log it)
            return True, ""
    
    @classmethod
    def scan_for_malicious_content(cls, file_path: Path) -> Tuple[bool, str]:
        """
        Scan file for potentially malicious content
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        # Check for executable signatures
        with open(file_path, 'rb') as f:
            # Read first 4 bytes
            magic_bytes = f.read(4)
            
            # PE executable (Windows)
            if magic_bytes[:2] == b'MZ':
                return False, "Executable file detected (PE format)"
            
            # ELF executable (Linux)
            if magic_bytes[:4] == b'\x7fELF':
                return False, "Executable file detected (ELF format)"
            
            # Mach-O executable (macOS)
            if magic_bytes[:4] in [b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe']:
                return False, "Executable file detected (Mach-O format)"
            
            # Script with shebang
            if magic_bytes[:2] == b'#!':
                return False, "Script file with shebang detected"
        
        return True, ""
    
    @classmethod
    def compute_file_hash(cls, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic hash of file
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hex digest of file hash
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @classmethod
    def validate_file(
        cls,
        file_path: Path,
        original_filename: str,
        check_mime: bool = True,
        check_malicious: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive file validation
        
        Args:
            file_path: Path to uploaded file
            original_filename: Original filename from upload
            check_mime: Whether to check MIME type
            check_malicious: Whether to scan for malicious content
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate filename
        valid, error = cls.validate_filename(original_filename)
        if not valid:
            errors.append(f"Filename: {error}")
        
        # Check file exists
        if not file_path.exists():
            errors.append("File does not exist")
            return False, errors
        
        # Validate file size
        file_size = file_path.stat().st_size
        valid, error = cls.validate_file_size(file_size)
        if not valid:
            errors.append(f"Size: {error}")
        
        # Validate MIME type
        if check_mime:
            valid, error = cls.validate_mime_type(file_path)
            if not valid:
                errors.append(f"Type: {error}")
        
        # Scan for malicious content
        if check_malicious:
            valid, error = cls.scan_for_malicious_content(file_path)
            if not valid:
                errors.append(f"Security: {error}")
        
        return len(errors) == 0, errors


class InputSanitizer:
    """
    Sanitize user inputs to prevent injection attacks
    """
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize file path to prevent directory traversal
        
        Args:
            path: User-provided path
            
        Returns:
            Sanitized path (relative, no traversal)
        """
        # Remove any absolute path indicators
        path = path.lstrip('/')
        path = path.lstrip('\\')
        
        # Remove drive letters (Windows)
        path = re.sub(r'^[A-Za-z]:', '', path)
        
        # Resolve path and ensure it doesn't escape
        try:
            clean_path = Path(path).resolve()
            # Remove any .. components
            parts = []
            for part in clean_path.parts:
                if part != '..' and part != '.':
                    parts.append(part)
            
            return str(Path(*parts)) if parts else "."
        
        except Exception:
            # If resolution fails, return safe default
            return "."
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """
        Sanitize text input for display/storage
        
        Args:
            text: User input text
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        # Truncate to max length
        text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove other control characters except newlines and tabs
        text = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        return text.strip()
    
    @staticmethod
    def sanitize_email(email: str) -> Tuple[bool, str]:
        """
        Validate and sanitize email address
        
        Args:
            email: Email address
            
        Returns:
            Tuple of (is_valid, sanitized_email)
        """
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        email = email.strip().lower()
        
        if not re.match(email_pattern, email):
            return False, ""
        
        if len(email) > 254:  # RFC 5321
            return False, ""
        
        return True, email
    
    @staticmethod
    def sanitize_dataset_name(name: str) -> str:
        """
        Sanitize dataset name for safe filesystem use
        
        Args:
            name: Dataset name
            
        Returns:
            Sanitized name
        """
        # Remove dangerous characters
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
        
        # Collapse multiple underscores/spaces
        name = re.sub(r'[_\s]+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure not empty
        if not name:
            name = "unnamed_dataset"
        
        # Limit length
        name = name[:100]
        
        return name
    
    @staticmethod
    def sanitize_sql_like(pattern: str) -> str:
        """
        Sanitize pattern for SQL LIKE queries
        
        Args:
            pattern: Search pattern
            
        Returns:
            Escaped pattern
        """
        # Escape SQL LIKE special characters
        pattern = pattern.replace('\\', '\\\\')
        pattern = pattern.replace('%', '\\%')
        pattern = pattern.replace('_', '\\_')
        pattern = pattern.replace('[', '\\[')
        pattern = pattern.replace(']', '\\]')
        
        return pattern


class RateLimiter:
    """
    Simple in-memory rate limiter for API endpoints
    """
    
    def __init__(self):
        """Initialize rate limiter"""
        self._requests: Dict[str, List[float]] = {}
    
    def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit
        
        Args:
            identifier: Unique identifier (e.g., IP address, user ID)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, seconds_until_retry)
        """
        import time
        
        current_time = time.time()
        
        # Initialize if first request
        if identifier not in self._requests:
            self._requests[identifier] = []
        
        # Remove expired requests
        cutoff_time = current_time - window_seconds
        self._requests[identifier] = [
            req_time for req_time in self._requests[identifier]
            if req_time > cutoff_time
        ]
        
        # Check if under limit
        if len(self._requests[identifier]) < max_requests:
            self._requests[identifier].append(current_time)
            return True, None
        
        # Calculate retry time
        oldest_request = min(self._requests[identifier])
        retry_after = int(oldest_request + window_seconds - current_time) + 1
        
        return False, retry_after
    
    def clear(self, identifier: str):
        """Clear rate limit for identifier"""
        if identifier in self._requests:
            del self._requests[identifier]


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    
    return _rate_limiter
