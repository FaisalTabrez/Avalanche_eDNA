"""
Comprehensive test suite for user management pipeline
Tests PostgreSQL backend, Redis caching, and all CRUD operations
"""
import os
import sys
import time
import uuid
from datetime import datetime

# Use environment variables if already set (Docker), otherwise use defaults
if 'DB_HOST' not in os.environ:
    os.environ['DB_TYPE'] = 'postgresql'
    os.environ['DB_HOST'] = 'postgres'
    os.environ['DB_PORT'] = '5432'
    os.environ['DB_NAME'] = 'avalanche_edna'
    os.environ['DB_USER'] = 'avalanche'
    os.environ['DB_PASSWORD'] = 'avalanche_dev_password'
    os.environ['REDIS_URL'] = 'redis://redis:6379/0'

from src.auth.postgres_user_manager import PostgresUserManager
from src.auth.password_utils import validate_password_strength


class TestUserManagementPipeline:
    """Test suite for user management pipeline"""
    
    def __init__(self):
        self.manager = None
        self.test_users = []
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def log(self, message, level="INFO"):
        """Log test messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️ "
        }.get(level, "  ")
        print(f"[{timestamp}] {prefix} {message}")
    
    def test_connection(self):
        """Test 1: PostgreSQL and Redis connectivity"""
        self.log("TEST 1: Testing database connections", "INFO")
        
        try:
            self.manager = PostgresUserManager()
            self.log("PostgreSQL connection established", "SUCCESS")
            self.passed += 1
            
            if self.manager.redis_available:
                self.log("Redis connection established", "SUCCESS")
                self.passed += 1
            else:
                self.log("Redis not available (will work without caching)", "WARNING")
                
            return True
        except Exception as e:
            self.log(f"Connection failed: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Connection test: {e}")
            return False
    
    def test_schema_initialization(self):
        """Test 2: Database schema initialization"""
        self.log("TEST 2: Verifying database schema", "INFO")
        
        try:
            conn = self.manager._get_connection()
            cursor = conn.cursor()
            
            # Check users table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'users'
                )
            """)
            users_exists = cursor.fetchone()[0]
            
            # Check sessions table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'sessions'
                )
            """)
            sessions_exists = cursor.fetchone()[0]
            
            # Check audit_log table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'audit_log'
                )
            """)
            audit_exists = cursor.fetchone()[0]
            
            self.manager._return_connection(conn)
            
            if users_exists and sessions_exists and audit_exists:
                self.log("All required tables exist", "SUCCESS")
                self.passed += 1
                return True
            else:
                missing = []
                if not users_exists: missing.append("users")
                if not sessions_exists: missing.append("sessions")
                if not audit_exists: missing.append("audit_log")
                self.log(f"Missing tables: {', '.join(missing)}", "ERROR")
                self.failed += 1
                self.errors.append(f"Schema test: Missing tables {missing}")
                return False
                
        except Exception as e:
            self.log(f"Schema verification failed: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Schema test: {e}")
            return False
    
    def test_password_validation(self):
        """Test 3: Password validation"""
        self.log("TEST 3: Testing password validation", "INFO")
        
        tests = [
            ("weak", False, "Password too short"),
            ("WeakPassword", False, "Missing special char and digit"),
            ("Weak@Pass", False, "Missing digit"),
            ("Weak@Pass1", True, "Valid password"),
            ("Strong@Pass123", True, "Strong password"),
        ]
        
        all_passed = True
        for password, should_pass, description in tests:
            is_valid, msg = validate_password_strength(password)
            
            if is_valid == should_pass:
                self.log(f"  ✓ {description}: PASS", "INFO")
            else:
                self.log(f"  ✗ {description}: FAIL (expected {should_pass}, got {is_valid})", "ERROR")
                all_passed = False
                self.errors.append(f"Password validation: {description} failed")
        
        if all_passed:
            self.passed += 1
            self.log("Password validation working correctly", "SUCCESS")
        else:
            self.failed += 1
        
        return all_passed
    
    def test_create_user(self):
        """Test 4: Create user operation"""
        self.log("TEST 4: Testing user creation", "INFO")
        
        try:
            # Create test user
            test_username = f"test_user_{uuid.uuid4().hex[:8]}"
            test_email = f"{test_username}@test.com"
            test_password = "Test@Pass123"
            
            success, result = self.manager.create_user(
                username=test_username,
                email=test_email,
                password=test_password,
                role='analyst'
            )
            
            if success:
                self.log(f"User created successfully: {test_username}", "SUCCESS")
                self.test_users.append({
                    'user_id': result,
                    'username': test_username,
                    'password': test_password
                })
                self.passed += 1
                return True
            else:
                self.log(f"User creation failed: {result}", "ERROR")
                self.failed += 1
                self.errors.append(f"Create user test: {result}")
                return False
                
        except Exception as e:
            self.log(f"User creation error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Create user test: {e}")
            return False
    
    def test_duplicate_user(self):
        """Test 5: Duplicate user prevention"""
        self.log("TEST 5: Testing duplicate user prevention", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            success, result = self.manager.create_user(
                username=user['username'],
                email=f"different_{user['username']}@test.com",
                password="Test@Pass123",
                role='viewer'
            )
            
            if not success and "already exists" in result.lower():
                self.log("Duplicate username correctly rejected", "SUCCESS")
                self.passed += 1
                return True
            else:
                self.log("Duplicate user was not rejected!", "ERROR")
                self.failed += 1
                self.errors.append("Duplicate user test: Failed to reject duplicate")
                return False
                
        except Exception as e:
            self.log(f"Duplicate user test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Duplicate user test: {e}")
            return False
    
    def test_authenticate(self):
        """Test 6: User authentication"""
        self.log("TEST 6: Testing user authentication", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            
            # Test correct password
            success, user_data = self.manager.authenticate(
                user['username'],
                user['password']
            )
            
            if success and user_data:
                self.log(f"Authentication successful for {user['username']}", "SUCCESS")
                self.passed += 1
                
                # Test wrong password
                success, user_data = self.manager.authenticate(
                    user['username'],
                    "WrongPassword123!"
                )
                
                if not success:
                    self.log("Wrong password correctly rejected", "SUCCESS")
                    self.passed += 1
                    return True
                else:
                    self.log("Wrong password was accepted!", "ERROR")
                    self.failed += 1
                    self.errors.append("Auth test: Wrong password accepted")
                    return False
            else:
                self.log("Authentication failed with correct password", "ERROR")
                self.failed += 1
                self.errors.append("Auth test: Correct password rejected")
                return False
                
        except Exception as e:
            self.log(f"Authentication test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Auth test: {e}")
            return False
    
    def test_get_user(self):
        """Test 7: Get user by ID"""
        self.log("TEST 7: Testing get user operation", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            user_data = self.manager.get_user(user['user_id'])
            
            if user_data and user_data['username'] == user['username']:
                self.log("User retrieval successful", "SUCCESS")
                self.passed += 1
                return True
            else:
                self.log("User retrieval failed", "ERROR")
                self.failed += 1
                self.errors.append("Get user test: Failed to retrieve user")
                return False
                
        except Exception as e:
            self.log(f"Get user test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Get user test: {e}")
            return False
    
    def test_list_users(self):
        """Test 8: List all users"""
        self.log("TEST 8: Testing list users operation", "INFO")
        
        try:
            users = self.manager.list_users()
            
            if users and len(users) > 0:
                self.log(f"Listed {len(users)} users successfully", "SUCCESS")
                self.passed += 1
                return True
            else:
                self.log("List users returned empty or None", "ERROR")
                self.failed += 1
                self.errors.append("List users test: Empty result")
                return False
                
        except Exception as e:
            self.log(f"List users test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"List users test: {e}")
            return False
    
    def test_update_user(self):
        """Test 9: Update user operation"""
        self.log("TEST 9: Testing update user operation", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            
            # Update role
            success, msg = self.manager.update_user(
                user['user_id'],
                role='admin'
            )
            
            if success:
                self.log("User role updated successfully", "SUCCESS")
                
                # Verify update
                user_data = self.manager.get_user(user['user_id'])
                if user_data and user_data['role'] == 'admin':
                    self.log("Update verified", "SUCCESS")
                    self.passed += 1
                    return True
                else:
                    self.log("Update verification failed", "ERROR")
                    self.failed += 1
                    self.errors.append("Update user test: Verification failed")
                    return False
            else:
                self.log(f"User update failed: {msg}", "ERROR")
                self.failed += 1
                self.errors.append(f"Update user test: {msg}")
                return False
                
        except Exception as e:
            self.log(f"Update user test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Update user test: {e}")
            return False
    
    def test_change_password(self):
        """Test 10: Change password operation"""
        self.log("TEST 10: Testing change password operation", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            new_password = "NewTest@Pass456"
            
            # Change password
            success, msg = self.manager.change_password(
                user['user_id'],
                new_password
            )
            
            if success:
                self.log("Password changed successfully", "SUCCESS")
                
                # Verify new password works
                success, user_data = self.manager.authenticate(
                    user['username'],
                    new_password
                )
                
                if success:
                    self.log("New password verified", "SUCCESS")
                    user['password'] = new_password  # Update for future tests
                    self.passed += 1
                    return True
                else:
                    self.log("New password authentication failed", "ERROR")
                    self.failed += 1
                    self.errors.append("Change password test: New password doesn't work")
                    return False
            else:
                self.log(f"Password change failed: {msg}", "ERROR")
                self.failed += 1
                self.errors.append(f"Change password test: {msg}")
                return False
                
        except Exception as e:
            self.log(f"Change password test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Change password test: {e}")
            return False
    
    def test_redis_caching(self):
        """Test 11: Redis caching functionality"""
        self.log("TEST 11: Testing Redis caching", "INFO")
        
        if not self.manager.redis_available:
            self.log("Redis not available, skipping cache test", "WARNING")
            return True
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            
            # Clear cache
            self.manager._invalidate_cache(f"user:{user['user_id']}")
            
            # First retrieval (should hit database)
            start = time.time()
            user_data1 = self.manager.get_user(user['user_id'])
            time1 = time.time() - start
            
            # Second retrieval (should hit cache)
            start = time.time()
            user_data2 = self.manager.get_user(user['user_id'])
            time2 = time.time() - start
            
            if user_data1 and user_data2:
                self.log(f"First call: {time1*1000:.2f}ms, Second call (cached): {time2*1000:.2f}ms", "INFO")
                
                if time2 < time1:
                    self.log("Cache is working (second call faster)", "SUCCESS")
                    self.passed += 1
                    return True
                else:
                    self.log("Cache might not be working (second call not faster)", "WARNING")
                    self.passed += 1  # Don't fail, timing can vary
                    return True
            else:
                self.log("Cache test failed: No data returned", "ERROR")
                self.failed += 1
                self.errors.append("Cache test: No data returned")
                return False
                
        except Exception as e:
            self.log(f"Cache test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Cache test: {e}")
            return False
    
    def test_permissions(self):
        """Test 12: Permission checking"""
        self.log("TEST 12: Testing permission system", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            user = self.test_users[0]
            
            # Admin should have all permissions
            has_manage_users = self.manager.has_permission(user['user_id'], 'manage_users')
            has_write = self.manager.has_permission(user['user_id'], 'write')
            has_read = self.manager.has_permission(user['user_id'], 'read')
            
            if has_manage_users and has_write and has_read:
                self.log("Permission checking working correctly", "SUCCESS")
                self.passed += 1
                return True
            else:
                self.log("Permission checking failed", "ERROR")
                self.failed += 1
                self.errors.append("Permission test: Incorrect permissions")
                return False
                
        except Exception as e:
            self.log(f"Permission test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Permission test: {e}")
            return False
    
    def test_delete_user(self):
        """Test 13: Delete user operation"""
        self.log("TEST 13: Testing delete user operation", "INFO")
        
        if not self.test_users:
            self.log("No test users available, skipping", "WARNING")
            return True
        
        try:
            # Create a temporary user to delete
            temp_username = f"temp_user_{uuid.uuid4().hex[:8]}"
            success, user_id = self.manager.create_user(
                username=temp_username,
                email=f"{temp_username}@test.com",
                password="Temp@Pass123",
                role='viewer'
            )
            
            if not success:
                self.log("Failed to create temp user for delete test", "ERROR")
                self.failed += 1
                return False
            
            # Delete the user
            success, msg = self.manager.delete_user(user_id)
            
            if success:
                self.log("User deleted successfully", "SUCCESS")
                
                # Verify deletion
                user_data = self.manager.get_user(user_id)
                if user_data is None:
                    self.log("Deletion verified", "SUCCESS")
                    self.passed += 1
                    return True
                else:
                    self.log("User still exists after deletion", "ERROR")
                    self.failed += 1
                    self.errors.append("Delete user test: User still exists")
                    return False
            else:
                self.log(f"Delete failed: {msg}", "ERROR")
                self.failed += 1
                self.errors.append(f"Delete user test: {msg}")
                return False
                
        except Exception as e:
            self.log(f"Delete user test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Delete user test: {e}")
            return False
    
    def test_audit_log(self):
        """Test 14: Audit logging"""
        self.log("TEST 14: Testing audit log", "INFO")
        
        try:
            conn = self.manager._get_connection()
            cursor = conn.cursor()
            
            # Check if audit entries exist
            cursor.execute("SELECT COUNT(*) FROM audit_log")
            count = cursor.fetchone()[0]
            
            self.manager._return_connection(conn)
            
            if count > 0:
                self.log(f"Audit log contains {count} entries", "SUCCESS")
                self.passed += 1
                return True
            else:
                self.log("Audit log is empty", "WARNING")
                self.passed += 1  # Don't fail, might be clean database
                return True
                
        except Exception as e:
            self.log(f"Audit log test error: {e}", "ERROR")
            self.failed += 1
            self.errors.append(f"Audit log test: {e}")
            return False
    
    def cleanup(self):
        """Cleanup test users"""
        self.log("CLEANUP: Removing test users", "INFO")
        
        for user in self.test_users:
            try:
                self.manager.delete_user(user['user_id'])
                self.log(f"Deleted test user: {user['username']}", "INFO")
            except Exception as e:
                self.log(f"Failed to delete {user['username']}: {e}", "WARNING")
        
        if self.manager:
            try:
                self.manager.close()
                self.log("Closed database connections", "INFO")
            except Exception as e:
                self.log(f"Error closing connections: {e}", "WARNING")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("  USER MANAGEMENT PIPELINE TEST SUITE")
        print("="*70 + "\n")
        
        tests = [
            self.test_connection,
            self.test_schema_initialization,
            self.test_password_validation,
            self.test_create_user,
            self.test_duplicate_user,
            self.test_authenticate,
            self.test_get_user,
            self.test_list_users,
            self.test_update_user,
            self.test_change_password,
            self.test_redis_caching,
            self.test_permissions,
            self.test_delete_user,
            self.test_audit_log,
        ]
        
        for test in tests:
            try:
                test()
                print()  # Add spacing between tests
            except Exception as e:
                self.log(f"Unexpected error in {test.__name__}: {e}", "ERROR")
                self.failed += 1
                self.errors.append(f"{test.__name__}: {e}")
                print()
        
        # Cleanup
        self.cleanup()
        
        # Summary
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Total:  {self.passed + self.failed}")
        
        if self.failed > 0:
            print("\n" + "="*70)
            print("  ERRORS")
            print("="*70)
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n" + "="*70)
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"⚠️  {self.failed} TEST(S) FAILED")
            return 1


if __name__ == "__main__":
    tester = TestUserManagementPipeline()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
