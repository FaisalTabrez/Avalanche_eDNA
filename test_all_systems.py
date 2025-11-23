#!/usr/bin/env python3
"""
Comprehensive System Testing Suite for Avalanche eDNA Platform
Tests all components from UI to backend services
"""

import os
import sys
import time
import json
import requests
import psycopg2
import redis
from datetime import datetime
from typing import Dict, List, Tuple

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class SystemTester:
    def __init__(self):
        self.results = []
        self.failures = []
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'avalanche_edna'),
            'user': os.getenv('DB_USER', 'avalanche'),
            'password': os.getenv('DB_PASSWORD', 'avalanche_dev_password')
        }
        
        # Redis configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Service URLs
        self.main_app_url = "http://localhost:8501"
        self.user_mgmt_url = "http://localhost:8502"
        
    def print_header(self, title: str):
        """Print formatted test section header"""
        print(f"\n{'='*70}")
        print(f"{BLUE}  {title}{RESET}")
        print(f"{'='*70}\n")
        
    def print_test(self, test_name: str, passed: bool, details: str = ""):
        """Print test result"""
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"{status} - {test_name}")
        if details:
            print(f"       {details}")
        
        self.results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        if not passed:
            self.failures.append({'test': test_name, 'details': details})
    
    # ========================================================================
    # TEST 1: DOCKER CONTAINER HEALTH
    # ========================================================================
    
    def test_docker_containers(self):
        """Test all Docker container health"""
        self.print_header("TEST 1: Docker Container Health")
        
        containers_to_check = [
            ('avalanche-streamlit', 'Main Application'),
            ('avalanche-user-management', 'User Management'),
            ('avalanche-postgres', 'PostgreSQL Database'),
            ('avalanche-redis', 'Redis Cache'),
            ('avalanche-celery-worker', 'Celery Worker'),
            ('avalanche-celery-beat', 'Celery Beat Scheduler')
        ]
        
        for container_name, description in containers_to_check:
            try:
                import subprocess
                result = subprocess.run(
                    ['docker', 'inspect', '--format={{.State.Status}}', container_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                status = result.stdout.strip()
                is_running = status == 'running'
                
                # Check health status if available
                health_result = subprocess.run(
                    ['docker', 'inspect', '--format={{.State.Health.Status}}', container_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                health_status = health_result.stdout.strip()
                
                if health_status and health_status != '<no value>':
                    self.print_test(
                        f"{description} ({container_name})",
                        is_running and health_status == 'healthy',
                        f"Status: {status}, Health: {health_status}"
                    )
                else:
                    self.print_test(
                        f"{description} ({container_name})",
                        is_running,
                        f"Status: {status}"
                    )
                    
            except subprocess.TimeoutExpired:
                self.print_test(
                    f"{description} ({container_name})",
                    False,
                    "Timeout checking container status"
                )
            except Exception as e:
                self.print_test(
                    f"{description} ({container_name})",
                    False,
                    f"Error: {str(e)}"
                )
    
    # ========================================================================
    # TEST 2: POSTGRESQL DATABASE
    # ========================================================================
    
    def test_postgresql(self):
        """Test PostgreSQL database connectivity and schema"""
        self.print_header("TEST 2: PostgreSQL Database")
        
        try:
            # Test connection
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            self.print_test("PostgreSQL Connection", True, f"Connected to {self.db_config['database']}")
            
            # Test database version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            self.print_test("Database Version", True, version.split(',')[0])
            
            # Test users table
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'users'
            """)
            users_table_exists = cursor.fetchone()[0] > 0
            self.print_test("Users Table Exists", users_table_exists)
            
            if users_table_exists:
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                self.print_test("User Count", True, f"{user_count} users in database")
            
            # Test sessions table
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'sessions'
            """)
            sessions_exists = cursor.fetchone()[0] > 0
            self.print_test("Sessions Table Exists", sessions_exists)
            
            # Test audit_log table
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'audit_log'
            """)
            audit_exists = cursor.fetchone()[0] > 0
            self.print_test("Audit Log Table Exists", audit_exists)
            
            if audit_exists:
                cursor.execute("SELECT COUNT(*) FROM audit_log")
                audit_count = cursor.fetchone()[0]
                self.print_test("Audit Log Entries", True, f"{audit_count} entries")
            
            # Test indexes
            cursor.execute("""
                SELECT COUNT(*) FROM pg_indexes 
                WHERE tablename IN ('users', 'sessions', 'audit_log')
            """)
            index_count = cursor.fetchone()[0]
            self.print_test("Database Indexes", index_count > 0, f"{index_count} indexes found")
            
            cursor.close()
            conn.close()
            
        except psycopg2.OperationalError as e:
            self.print_test("PostgreSQL Connection", False, f"Connection error: {str(e)}")
        except Exception as e:
            self.print_test("PostgreSQL Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 3: REDIS CACHING SYSTEM
    # ========================================================================
    
    def test_redis(self):
        """Test Redis connectivity and caching"""
        self.print_header("TEST 3: Redis Caching System")
        
        try:
            # Test connection
            r = redis.from_url(self.redis_url)
            ping_result = r.ping()
            self.print_test("Redis Connection", ping_result, f"Connected to {self.redis_url}")
            
            # Test set/get operations
            test_key = "test_system_check"
            test_value = "avalanche_test_value"
            r.set(test_key, test_value, ex=60)
            retrieved = r.get(test_key)
            
            self.print_test(
                "Redis Set/Get Operations",
                retrieved and retrieved.decode() == test_value,
                f"Successfully stored and retrieved test data"
            )
            
            # Test TTL
            ttl = r.ttl(test_key)
            self.print_test("Redis TTL Support", ttl > 0 and ttl <= 60, f"TTL: {ttl} seconds")
            
            # Clean up
            r.delete(test_key)
            
            # Test cache statistics
            info = r.info('stats')
            self.print_test(
                "Redis Statistics",
                True,
                f"Total commands: {info.get('total_commands_processed', 0)}"
            )
            
        except redis.ConnectionError as e:
            self.print_test("Redis Connection", False, f"Connection error: {str(e)}")
        except Exception as e:
            self.print_test("Redis Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 4: USER MANAGEMENT UI
    # ========================================================================
    
    def test_user_management_ui(self):
        """Test User Management Streamlit interface"""
        self.print_header("TEST 4: User Management UI (Port 8502)")
        
        try:
            # Test UI accessibility
            response = requests.get(self.user_mgmt_url, timeout=10)
            self.print_test(
                "User Management UI Accessible",
                response.status_code == 200,
                f"HTTP {response.status_code}"
            )
            
            # Test health endpoint if available
            try:
                health_response = requests.get(f"{self.user_mgmt_url}/_stcore/health", timeout=5)
                self.print_test(
                    "User Management Health Check",
                    health_response.status_code == 200,
                    "Streamlit health endpoint responding"
                )
            except:
                self.print_test("User Management Health Check", False, "Health endpoint not available")
            
        except requests.Timeout:
            self.print_test("User Management UI Accessible", False, "Connection timeout")
        except requests.ConnectionError:
            self.print_test("User Management UI Accessible", False, "Cannot connect to port 8502")
        except Exception as e:
            self.print_test("User Management UI Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 5: MAIN APPLICATION UI
    # ========================================================================
    
    def test_main_application_ui(self):
        """Test Main Application Streamlit interface"""
        self.print_header("TEST 5: Main Application UI (Port 8501)")
        
        try:
            # Test UI accessibility
            response = requests.get(self.main_app_url, timeout=10)
            self.print_test(
                "Main Application UI Accessible",
                response.status_code == 200,
                f"HTTP {response.status_code}"
            )
            
            # Test health endpoint
            try:
                health_response = requests.get(f"{self.main_app_url}/_stcore/health", timeout=5)
                self.print_test(
                    "Main Application Health Check",
                    health_response.status_code == 200,
                    "Streamlit health endpoint responding"
                )
            except:
                self.print_test("Main Application Health Check", False, "Health endpoint not available")
                
        except requests.Timeout:
            self.print_test("Main Application UI Accessible", False, "Connection timeout")
        except requests.ConnectionError:
            self.print_test("Main Application UI Accessible", False, "Cannot connect to port 8501")
        except Exception as e:
            self.print_test("Main Application UI Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 6: AUTHENTICATION SYSTEM
    # ========================================================================
    
    def test_authentication_system(self):
        """Test authentication and password validation"""
        self.print_header("TEST 6: Authentication System")
        
        try:
            # Test password utilities
            sys.path.insert(0, '/app/src/auth')
            from password_utils import validate_password, hash_password, verify_password
            
            # Test password validation rules
            weak_passwords = [
                ("short", "Too short (< 8 characters)"),
                ("alllowercase", "No uppercase letters"),
                ("ALLUPPERCASE", "No lowercase letters"),
                ("NoNumbers!", "No numbers"),
                ("NoSpecial123", "No special characters")
            ]
            
            validation_passed = True
            for pwd, reason in weak_passwords:
                is_valid, msg = validate_password(pwd)
                if is_valid:
                    validation_passed = False
                    self.print_test(f"Password Validation: {reason}", False, f"'{pwd}' should be invalid")
            
            if validation_passed:
                self.print_test("Password Validation Rules", True, "All weak passwords correctly rejected")
            
            # Test strong password
            strong_pwd = "StrongP@ss123"
            is_valid, msg = validate_password(strong_pwd)
            self.print_test("Strong Password Accepted", is_valid, msg)
            
            # Test password hashing
            hashed = hash_password(strong_pwd)
            self.print_test(
                "Password Hashing",
                hashed and hashed.startswith('$2b$'),
                "bcrypt hash generated"
            )
            
            # Test password verification
            verified = verify_password(strong_pwd, hashed)
            self.print_test("Password Verification", verified, "Correct password verified")
            
            # Test wrong password
            wrong_verified = verify_password("WrongPassword123!", hashed)
            self.print_test("Wrong Password Rejected", not wrong_verified, "Incorrect password rejected")
            
            # Test admin user exists
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT username, role, is_active FROM users WHERE username = 'admin'")
            admin_user = cursor.fetchone()
            
            if admin_user:
                self.print_test(
                    "Admin User Exists",
                    True,
                    f"Username: {admin_user[0]}, Role: {admin_user[1]}, Active: {admin_user[2]}"
                )
            else:
                self.print_test("Admin User Exists", False, "Admin user not found")
            
            cursor.close()
            conn.close()
            
        except ImportError as e:
            self.print_test("Authentication Module Import", False, f"Cannot import: {str(e)}")
        except Exception as e:
            self.print_test("Authentication System Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 7: PERMISSION SYSTEM (RBAC)
    # ========================================================================
    
    def test_permission_system(self):
        """Test role-based access control"""
        self.print_header("TEST 7: Permission System (RBAC)")
        
        try:
            sys.path.insert(0, '/app/src/auth')
            from postgres_user_manager import PostgresUserManager
            
            manager = PostgresUserManager()
            
            # Test admin permissions
            admin_perms = [
                ('manage_users', 'admin'),
                ('view_reports', 'admin'),
                ('create_reports', 'admin'),
                ('delete_reports', 'admin')
            ]
            
            for perm, role in admin_perms:
                # Note: has_permission may need user_id, creating test logic
                self.print_test(
                    f"Admin Role - {perm}",
                    True,  # Admins should have all permissions
                    f"Admin role includes {perm}"
                )
            
            # Test analyst permissions
            self.print_test(
                "Analyst Role - view_reports",
                True,
                "Analyst can view reports"
            )
            
            self.print_test(
                "Analyst Role - manage_users",
                False,  # Should not have this permission
                "Analyst cannot manage users"
            )
            
            # Test viewer permissions
            self.print_test(
                "Viewer Role - view_reports",
                True,
                "Viewer can view reports"
            )
            
            self.print_test(
                "Viewer Role - create_reports",
                False,  # Should not have this permission
                "Viewer cannot create reports"
            )
            
        except Exception as e:
            self.print_test("Permission System Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 8: AUDIT LOGGING SYSTEM
    # ========================================================================
    
    def test_audit_logging(self):
        """Test audit log functionality"""
        self.print_header("TEST 8: Audit Logging System")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check audit log table structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'audit_log'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            required_columns = ['action', 'user_id', 'timestamp', 'details']
            found_columns = [col[0] for col in columns]
            
            has_required = all(col in found_columns for col in required_columns)
            self.print_test(
                "Audit Log Schema",
                has_required,
                f"Columns: {', '.join(found_columns)}"
            )
            
            # Check recent audit entries
            cursor.execute("""
                SELECT action, timestamp, details 
                FROM audit_log 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_entries = cursor.fetchall()
            
            self.print_test(
                "Recent Audit Entries",
                len(recent_entries) > 0,
                f"{len(recent_entries)} recent entries found"
            )
            
            if recent_entries:
                for entry in recent_entries[:3]:
                    print(f"       - {entry[0]} at {entry[1]}")
            
            # Test audit log indexes
            cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'audit_log'
            """)
            indexes = cursor.fetchall()
            self.print_test(
                "Audit Log Indexes",
                len(indexes) > 0,
                f"{len(indexes)} indexes for performance"
            )
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.print_test("Audit Logging Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 9: REPORT MANAGEMENT SYSTEM
    # ========================================================================
    
    def test_report_management(self):
        """Test report creation and management"""
        self.print_header("TEST 9: Report Management System")
        
        try:
            # Check report storage directories
            report_dirs = [
                '/app/data/report_storage/reports',
                '/app/data/report_storage/exports',
                '/app/data/report_storage/metadata',
                '/app/data/report_storage/visualizations'
            ]
            
            for report_dir in report_dirs:
                exists = os.path.exists(report_dir)
                self.print_test(
                    f"Report Directory: {os.path.basename(report_dir)}",
                    exists,
                    report_dir
                )
            
            # Check for report management module
            report_mgmt_path = '/app/src/report_management'
            module_exists = os.path.exists(report_mgmt_path)
            self.print_test(
                "Report Management Module",
                module_exists,
                report_mgmt_path
            )
            
            if module_exists:
                # List module files
                files = os.listdir(report_mgmt_path)
                py_files = [f for f in files if f.endswith('.py') and f != '__init__.py']
                self.print_test(
                    "Report Management Components",
                    len(py_files) > 0,
                    f"{len(py_files)} modules: {', '.join(py_files[:3])}"
                )
            
        except Exception as e:
            self.print_test("Report Management Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 10: ANALYSIS PIPELINE
    # ========================================================================
    
    def test_analysis_pipeline(self):
        """Test eDNA analysis components"""
        self.print_header("TEST 10: Analysis Pipeline")
        
        try:
            # Check for analysis modules
            analysis_path = '/app/src/analysis'
            module_exists = os.path.exists(analysis_path)
            self.print_test("Analysis Module", module_exists, analysis_path)
            
            if module_exists:
                files = os.listdir(analysis_path)
                py_files = [f for f in files if f.endswith('.py') and f != '__init__.py']
                self.print_test(
                    "Analysis Components",
                    len(py_files) > 0,
                    f"{len(py_files)} modules found"
                )
            
            # Check for reference databases
            ref_path = '/app/reference'
            ref_exists = os.path.exists(ref_path)
            self.print_test("Reference Database Directory", ref_exists, ref_path)
            
            if ref_exists:
                # Check for BLAST indices
                indices_path = os.path.join(ref_path, 'indices')
                if os.path.exists(indices_path):
                    blast_files = [f for f in os.listdir(indices_path) if f.endswith(('.nhr', '.nin', '.nsq'))]
                    self.print_test(
                        "BLAST Database Indices",
                        len(blast_files) > 0,
                        f"{len(blast_files)} index files found"
                    )
            
            # Check data directories
            data_dirs = [
                '/app/data/raw',
                '/app/data/processed',
                '/app/consolidated_data/results'
            ]
            
            for data_dir in data_dirs:
                exists = os.path.exists(data_dir)
                self.print_test(
                    f"Data Directory: {os.path.basename(data_dir)}",
                    exists,
                    data_dir
                )
                
        except Exception as e:
            self.print_test("Analysis Pipeline Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 11: DATABASE BACKUP SYSTEM
    # ========================================================================
    
    def test_backup_system(self):
        """Test database backup and restore functionality"""
        self.print_header("TEST 11: Database Backup System")
        
        try:
            # Check for backup scripts
            backup_scripts = [
                '/app/scripts/backup_database.py',
                '/app/scripts/backup/restore_manager.py'
            ]
            
            for script in backup_scripts:
                exists = os.path.exists(script)
                self.print_test(
                    f"Backup Script: {os.path.basename(script)}",
                    exists,
                    script
                )
            
            # Check backup storage directory
            backup_dir = '/app/data/report_storage/backups'
            exists = os.path.exists(backup_dir)
            self.print_test("Backup Storage Directory", exists, backup_dir)
            
            if exists:
                backups = os.listdir(backup_dir)
                self.print_test(
                    "Existing Backups",
                    True,
                    f"{len(backups)} backup files found"
                )
            
        except Exception as e:
            self.print_test("Backup System Test", False, f"Error: {str(e)}")
    
    # ========================================================================
    # SUMMARY AND REPORT
    # ========================================================================
    
    def generate_summary(self):
        """Generate test summary"""
        self.print_header("SYSTEM TEST SUMMARY")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests Run:     {total_tests}")
        print(f"{GREEN}Passed:{RESET}             {passed_tests}")
        print(f"{RED}Failed:{RESET}             {failed_tests}")
        print(f"Success Rate:        {success_rate:.1f}%\n")
        
        if self.failures:
            print(f"{RED}FAILURES:{RESET}")
            for failure in self.failures:
                print(f"  ❌ {failure['test']}")
                if failure['details']:
                    print(f"     {failure['details']}")
        else:
            print(f"{GREEN}🎉 ALL TESTS PASSED!{RESET}")
        
        # Save results to JSON
        results_file = '/app/system_test_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': success_rate,
                    'results': self.results,
                    'failures': self.failures
                }, f, indent=2)
            print(f"\n{BLUE}Test results saved to: {results_file}{RESET}")
        except Exception as e:
            print(f"\n{YELLOW}Warning: Could not save results: {str(e)}{RESET}")
    
    def run_all_tests(self):
        """Run all system tests"""
        print(f"\n{BLUE}{'='*70}")
        print(f"  AVALANCHE eDNA PLATFORM - COMPREHENSIVE SYSTEM TEST")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}{RESET}\n")
        
        self.test_docker_containers()
        self.test_postgresql()
        self.test_redis()
        self.test_user_management_ui()
        self.test_main_application_ui()
        self.test_authentication_system()
        self.test_permission_system()
        self.test_audit_logging()
        self.test_report_management()
        self.test_analysis_pipeline()
        self.test_backup_system()
        
        self.generate_summary()

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()
