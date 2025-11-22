# Security Documentation

This document describes the security features implemented in Avalanche for production deployment.

## Overview

Avalanche implements comprehensive security controls for:
- File upload validation
- Input sanitization
- Rate limiting
- Path traversal protection
- Session security
- Security event logging

## Security Architecture

### Security Module (`src/security/`)

The security module provides three main components:

#### 1. File Validator (`FileValidator`)

Validates all file uploads before processing.

**Features:**
- Filename validation (extension and dangerous patterns)
- File size limits (500MB default, configurable)
- MIME type detection using libmagic
- Malicious content scanning (PE/ELF/Mach-O executables, scripts)
- Cryptographic hashing (SHA256/MD5/SHA1)

**Allowed File Types:**
- FASTA: `.fasta`, `.fa`, `.fna`, `.ffn`, `.faa`, `.frn`
- FASTQ: `.fastq`, `.fq`
- Compressed: `.gz`, `.bz2`, `.zip`
- Data: `.csv`, `.tsv`, `.txt`, `.json`, `.yaml`, `.yml`
- Bioinformatics: `.gbk`, `.genbank`, `.sam`, `.bam`, `.vcf`

**Usage:**
```python
from src.security import FileValidator

# Validate uploaded file
valid, error = FileValidator.validate_file(
    file_path="data/sample.fasta",
    check_mime=True,
    scan_malicious=True
)

if not valid:
    print(f"Validation failed: {error}")
```

**Configuration:**
Set `MAX_FILE_SIZE_MB` environment variable to override default 500MB limit.

#### 2. Input Sanitizer (`InputSanitizer`)

Sanitizes all user inputs to prevent injection attacks.

**Features:**
- Path traversal prevention (removes `..`, absolute paths)
- Text sanitization (removes control characters, truncates)
- Email validation (RFC 5321 compliant)
- Dataset name sanitization (filesystem-safe)
- SQL LIKE wildcard escaping

**Usage:**
```python
from src.security import InputSanitizer

# Sanitize dataset name
safe_name = InputSanitizer.sanitize_dataset_name("My Dataset!@#")
# Returns: "My_Dataset"

# Sanitize path
safe_path = InputSanitizer.sanitize_path("/data/../etc/passwd")
# Returns: "data/etc/passwd"

# Validate email
is_valid = InputSanitizer.sanitize_email("user@example.com")
# Returns: True
```

#### 3. Rate Limiter (`RateLimiter`)

Prevents abuse through request rate limiting.

**Features:**
- Sliding window rate limiting
- Configurable limits per endpoint
- Session/IP-based tracking
- Automatic cleanup of expired entries

**Usage:**
```python
from src.security import get_rate_limiter

limiter = get_rate_limiter()

# Check if request is allowed (10 requests per hour)
if limiter.is_allowed('upload_file', max_requests=10, window_seconds=3600):
    process_upload()
else:
    print("Rate limit exceeded")
```

### Security Middleware

#### Decorators

**@rate_limit** - Apply rate limiting to functions
```python
from src.security import rate_limit

@rate_limit(max_requests=10, window_seconds=3600, identifier_func=lambda: "upload")
def upload_file():
    # Process upload
    pass
```

**@validate_file_upload** - Validate file uploads automatically
```python
from src.security import validate_file_upload

@validate_file_upload(
    check_mime=True,
    scan_malicious=True,
    file_param_name='uploaded_file'
)
def process_upload(uploaded_file):
    # File is already validated
    pass
```

**@sanitize_input** - Sanitize text inputs
```python
from src.security import sanitize_input

@sanitize_input(params=['dataset_name'])
def create_dataset(dataset_name: str):
    # dataset_name is already sanitized
    pass
```

#### Session Security

Initialize session security in Streamlit:
```python
from src.security import initialize_session_security

# Call before state initialization
initialize_session_security()
```

**Features:**
- UUID session IDs
- Creation timestamps
- Request counting
- Session expiration tracking

#### Path Traversal Protection

Validate paths before file operations:
```python
from src.security import check_path_traversal, validate_dataset_path

# Check if path is within base directory
if not check_path_traversal("/data/uploads/file.txt", "/data"):
    raise ValueError("Path traversal detected")

# Construct safe dataset path
dataset_path = validate_dataset_path("My_Dataset", "/data/datasets")
# Returns: /data/datasets/My_Dataset (validated)
```

### Security Headers

For FastAPI applications, use `SecurityHeaders` middleware:
```python
from src.security import SecurityHeaders
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(SecurityHeaders)
```

**Headers Applied:**
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'`
- `Permissions-Policy: geolocation=(), camera=(), microphone=()`

## Security Event Logging

All security events are logged with structured data:

```python
from src.security import log_security_event

# Log security event
log_security_event('file_upload_rejected', {
    'filename': 'malicious.exe',
    'reason': 'executable_detected',
    'user_id': user.id
})
```

**Common Events:**
- `file_upload_rejected` - File upload failed validation
- `path_traversal_attempt` - Path traversal detected
- `rate_limit_exceeded` - Rate limit triggered
- `invalid_input` - Input sanitization failed
- `session_security_initialized` - Session created

## Implementation Examples

### File Upload with Validation

```python
import streamlit as st
from src.security import FileValidator, get_rate_limiter, log_security_event

uploaded_file = st.file_uploader("Upload File", type=['fasta'])

if uploaded_file:
    # Rate limiting
    if not get_rate_limiter().is_allowed('upload', max_requests=10, window_seconds=3600):
        st.error("Upload limit exceeded")
    else:
        # Validate file
        valid, error = FileValidator.validate_file(
            uploaded_file.name,
            check_mime=True,
            scan_malicious=True
        )
        
        if not valid:
            st.error(f"Invalid file: {error}")
            log_security_event('file_upload_rejected', {
                'filename': uploaded_file.name,
                'error': error
            })
        else:
            # Process file
            process_upload(uploaded_file)
```

### ZIP File Extraction with Safety Checks

```python
import zipfile
from pathlib import Path
from src.security import FileValidator, log_security_event

def extract_zip_safely(zip_path: Path, extract_to: Path):
    """Extract ZIP with security checks"""
    
    # Validate it's a ZIP
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Not a valid ZIP file")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Check for zip bombs
        uncompressed = sum(info.file_size for info in zip_ref.filelist)
        compressed = sum(info.compress_size for info in zip_ref.filelist)
        
        if compressed > 0 and (uncompressed / compressed) > 100:
            log_security_event('zip_bomb_detected', {'file': str(zip_path)})
            raise ValueError("Suspicious compression ratio")
        
        # Validate all member paths
        for member in zip_ref.namelist():
            member_path = extract_to / member
            try:
                member_path.resolve().relative_to(extract_to.resolve())
            except ValueError:
                log_security_event('zip_path_traversal', {
                    'file': str(zip_path),
                    'member': member
                })
                raise ValueError(f"Invalid path in archive: {member}")
        
        # Extract
        zip_ref.extractall(extract_to)
```

### Dataset Name Sanitization

```python
import streamlit as st
from src.security import InputSanitizer, log_security_event

dataset_name_input = st.text_input("Dataset Name")

# Sanitize
dataset_name = InputSanitizer.sanitize_dataset_name(dataset_name_input)

if dataset_name != dataset_name_input:
    st.warning(f"Name sanitized: '{dataset_name_input}' → '{dataset_name}'")
    log_security_event('dataset_name_sanitized', {
        'original': dataset_name_input,
        'sanitized': dataset_name
    })
```

## Best Practices

### File Uploads

1. **Always validate filenames** before processing
2. **Check file sizes** before reading content
3. **Verify MIME types** using libmagic (not just extensions)
4. **Scan for malicious content** (executables, scripts)
5. **Apply rate limiting** to upload endpoints
6. **Log security events** for rejected uploads

### Input Handling

1. **Sanitize all user inputs** before use
2. **Validate paths** before file operations
3. **Use parameterized queries** for database operations
4. **Escape SQL LIKE wildcards** when needed
5. **Validate emails** with RFC-compliant regex

### Path Operations

1. **Always check for path traversal** using `check_path_traversal()`
2. **Use `validate_dataset_path()`** to construct safe paths
3. **Never trust user-provided paths** without validation
4. **Resolve paths** to detect symlinks and relative components

### Rate Limiting

1. **Apply to all resource-intensive operations** (uploads, analysis, training)
2. **Use appropriate limits** per operation type
3. **Consider different limits** for authenticated vs anonymous users
4. **Log rate limit violations** for abuse detection

### ZIP Archives

1. **Validate ZIP structure** before extraction
2. **Check compression ratios** to detect zip bombs
3. **Validate member paths** to prevent path traversal
4. **Scan extracted files** for malicious content
5. **Set extraction size limits** to prevent resource exhaustion

## Security Testing

### Manual Testing

Test file upload validation:
```bash
# Test oversized file
dd if=/dev/zero of=large.fasta bs=1M count=600  # 600MB
# Upload should be rejected

# Test malicious filename
touch "../../../etc/passwd"
# Upload should be rejected

# Test executable disguised as FASTA
cp /bin/ls malicious.fasta
# Upload should be rejected (MIME type mismatch)
```

Test path traversal:
```python
from src.security import check_path_traversal

# Should fail
assert not check_path_traversal("/data/../etc/passwd", "/data")
assert not check_path_traversal("/etc/passwd", "/data")

# Should pass
assert check_path_traversal("/data/uploads/file.txt", "/data")
```

Test rate limiting:
```python
from src.security import get_rate_limiter

limiter = get_rate_limiter()

# Should allow first 10 requests
for i in range(10):
    assert limiter.is_allowed('test', max_requests=10, window_seconds=60)

# Should reject 11th request
assert not limiter.is_allowed('test', max_requests=10, window_seconds=60)
```

### Automated Testing

Create `tests/test_security.py`:
```python
import pytest
from src.security import FileValidator, InputSanitizer, RateLimiter

def test_file_validator():
    # Test valid filename
    valid, _ = FileValidator.validate_filename("test.fasta")
    assert valid
    
    # Test invalid filename
    valid, _ = FileValidator.validate_filename("../../../etc/passwd")
    assert not valid

def test_input_sanitizer():
    # Test path sanitization
    assert InputSanitizer.sanitize_path("/data/../etc") == "data/etc"
    
    # Test dataset name sanitization
    assert InputSanitizer.sanitize_dataset_name("My Dataset!@#") == "My_Dataset"

def test_rate_limiter():
    limiter = RateLimiter()
    
    # Should allow within limit
    for _ in range(5):
        assert limiter.is_allowed('test', max_requests=5, window_seconds=60)
    
    # Should reject over limit
    assert not limiter.is_allowed('test', max_requests=5, window_seconds=60)
```

Run tests:
```bash
pytest tests/test_security.py -v
```

## Dependencies

- `python-magic>=0.4.27` - MIME type detection
- `python-magic-bin>=0.4.14` - Windows libmagic binaries

Install:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

- `MAX_FILE_SIZE_MB` - Maximum file upload size in MB (default: 500)
- `LOG_LEVEL` - Logging level for security events (default: INFO)

### Security Settings

Edit `config/config.yaml`:
```yaml
security:
  max_file_size_mb: 500
  allowed_extensions:
    - .fasta
    - .fa
    - .fastq
    - .fq
    # ...
  rate_limits:
    upload: 10  # requests per hour
    analysis: 20  # requests per hour
    training: 5  # requests per hour
```

## Monitoring

Security events are logged to:
- Console (during development)
- Log files (`logs/security.log`)
- External monitoring (if configured)

**Monitor for:**
- High rate of rejected uploads
- Path traversal attempts
- Rate limit violations
- Suspicious file uploads

**Example log entry:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "event": "file_upload_rejected",
  "details": {
    "filename": "malicious.exe",
    "reason": "executable_detected",
    "user_id": "user123",
    "ip": "192.168.1.100"
  }
}
```

## Incident Response

If security event detected:

1. **Review logs** to understand scope
2. **Block malicious users** if necessary
3. **Update validation rules** if new attack vector
4. **Notify affected users** if data breach
5. **Document incident** for future reference

## References

- [OWASP File Upload Security](https://owasp.org/www-community/vulnerabilities/Unrestricted_File_Upload)
- [CWE-22: Path Traversal](https://cwe.mitre.org/data/definitions/22.html)
- [CWE-409: ZIP Bomb](https://cwe.mitre.org/data/definitions/409.html)
- [RFC 5321: SMTP Email](https://tools.ietf.org/html/rfc5321)

## Support

For security issues:
1. Do NOT create public GitHub issues
2. Email security concerns to: security@avalanche-project.org
3. Include detailed information about the vulnerability
4. Allow 90 days for response before public disclosure
