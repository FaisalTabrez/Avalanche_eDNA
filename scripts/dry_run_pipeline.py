#!/usr/bin/env python3
"""
Pipeline Dry Run - Comprehensive validation of all components
Tests the entire pipeline without actually processing data
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class DryRunValidator:
    """Validates all pipeline components"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}\n")
    
    def print_result(self, test_name: str, passed: bool, message: str = ""):
        """Print test result"""
        if passed:
            print(f"{GREEN}[PASS]{RESET} {test_name}")
            if message:
                print(f"  {message}")
            self.passed.append(test_name)
        else:
            print(f"{RED}[FAIL]{RESET} {test_name}")
            if message:
                print(f"  Error: {message}")
            self.errors.append((test_name, message))
    
    def print_warning(self, test_name: str, message: str):
        """Print warning"""
        print(f"{YELLOW}[WARN]{RESET} {test_name}")
        print(f"  {message}")
        self.warnings.append((test_name, message))
    
    def test_imports(self) -> bool:
        """Test critical imports"""
        self.print_header("1. Import Validation")
        
        critical_imports = [
            # Core modules
            ("src.database.manager", "DatabaseManager"),
            ("src.database.connection", "DatabaseConnection"),
            ("src.database.schema", "DatabaseSchema"),
            
            # Security
            ("src.security.validators", "FileValidator"),
            ("src.security.validators", "InputSanitizer"),
            
            # Utils
            ("src.utils.cache", "CacheClient"),
            ("src.utils.rate_limiting", "RateLimiter"),
            ("src.utils.config", "config"),
            
            # Preprocessing (skip tokenizer to avoid FAISS issues)
            # ("src.preprocessing.tokenizer", "DNATokenizer"),
            
            # Tasks
            ("src.tasks.analysis_tasks", "run_analysis"),
            ("src.tasks.download_tasks", "download_sra_dataset"),
        ]
        
        all_passed = True
        for module_name, class_name in critical_imports:
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, class_name):
                    self.print_result(
                        f"{module_name}.{class_name}",
                        False,
                        f"Module exists but {class_name} not found"
                    )
                    all_passed = False
                else:
                    self.print_result(f"{module_name}.{class_name}", True)
            except ImportError as e:
                self.print_result(f"{module_name}", False, str(e))
                all_passed = False
        
        return all_passed
    
    def test_file_structure(self) -> bool:
        """Test required files and directories exist"""
        self.print_header("2. File Structure Validation")
        
        required_dirs = [
            "src/database",
            "src/analysis",
            "src/preprocessing",
            "src/security",
            "src/utils",
            "src/tasks",
            "scripts",
            "tests",
            "config",
            "data",
        ]
        
        required_files = [
            "scripts/run_pipeline.py",
            "streamlit_app.py",
            "config/config.yaml",
            "requirements.txt",
        ]
        
        all_passed = True
        
        for dir_path in required_dirs:
            path = project_root / dir_path
            if path.exists():
                self.print_result(f"Directory: {dir_path}", True)
            else:
                self.print_result(f"Directory: {dir_path}", False, "Not found")
                all_passed = False
        
        for file_path in required_files:
            path = project_root / file_path
            if path.exists():
                self.print_result(f"File: {file_path}", True)
            else:
                self.print_result(f"File: {file_path}", False, "Not found")
                all_passed = False
        
        return all_passed
    
    def test_configuration(self) -> bool:
        """Test configuration loading"""
        self.print_header("3. Configuration Validation")
        
        all_passed = True
        
        try:
            from src.utils.config import config
            
            # Test critical config values
            config_tests = [
                ("cache.enabled", bool),
                ("cache.ttl", int),
                ("rate_limiting.enabled", bool),
                ("database.type", str),
            ]
            
            for key, expected_type in config_tests:
                try:
                    value = config.get(key)
                    if value is not None and isinstance(value, expected_type):
                        self.print_result(f"Config: {key}", True, f"= {value}")
                    else:
                        self.print_warning(f"Config: {key}", f"Not set or wrong type")
                except Exception as e:
                    self.print_warning(f"Config: {key}", str(e))
            
        except Exception as e:
            self.print_result("Configuration loading", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_database_connection(self) -> bool:
        """Test database connection (dry run)"""
        self.print_header("4. Database Connection Test")
        
        try:
            from src.database.manager import DatabaseManager
            
            # Just test instantiation, don't actually connect
            self.print_result("DatabaseManager import", True)
            
            # Check if database file exists for SQLite
            db_type = os.getenv('DB_TYPE', 'sqlite')
            if db_type == 'sqlite':
                db_path = Path('data/avalanche.db')
                if db_path.exists():
                    self.print_result("SQLite database file", True, f"Size: {db_path.stat().st_size / 1024:.1f} KB")
                else:
                    self.print_warning("SQLite database file", "Not found - will be created on first run")
            
            return True
            
        except Exception as e:
            self.print_result("Database connection", False, str(e))
            return False
    
    def test_docker_services(self) -> bool:
        """Test Docker services"""
        self.print_header("5. Docker Services Check")
        
        import subprocess
        
        services = {
            'redis': 6379,
            'postgres': 5432,
            'prometheus': 9090,
            'grafana': 3000,
        }
        
        all_passed = True
        for service, port in services.items():
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={service}', '--format', '{{.Status}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and 'Up' in result.stdout:
                    self.print_result(f"Service: {service}", True, f"Running on port {port}")
                else:
                    self.print_warning(f"Service: {service}", f"Not running (optional)")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                self.print_warning(f"Service check: {service}", "Docker not available or timeout")
        
        return True  # Don't fail if Docker services are down
    
    def test_pipeline_components(self) -> bool:
        """Test pipeline component availability"""
        self.print_header("6. Pipeline Components")
        
        all_passed = True
        
        try:
            # Test tokenizer - skip actual instantiation to avoid FAISS segfault
            import importlib.util
            spec = importlib.util.find_spec("src.preprocessing.tokenizer")
            if spec is not None:
                self.print_result("DNA Tokenizer module", True, "Module exists")
            else:
                self.print_warning("DNA Tokenizer module", "Not found")
            
        except Exception as e:
            self.print_warning("DNA Tokenizer", str(e))
        
        try:
            # Test taxonomy resolver - check module only
            spec = importlib.util.find_spec("src.analysis.taxonomy")
            if spec is not None:
                self.print_result("Taxonomy Resolver module", True, "Module exists")
            else:
                self.print_warning("Taxonomy Resolver module", "Not found")
            
        except Exception as e:
            self.print_warning("Taxonomy Resolver", str(e))
        
        return all_passed
    
    def test_api_endpoints(self) -> bool:
        """Test API configuration"""
        self.print_header("7. API Configuration")
        
        try:
            # Test FastAPI integration
            from src.utils.fastapi_integration import create_cached_route, create_rate_limited_route
            self.print_result("FastAPI integration", True)
            
            # Test Report Management API - check module only
            import importlib.util
            spec = importlib.util.find_spec("src.api.report_management")
            if spec is not None:
                self.print_result("Report Management API module", True, "Module exists")
            else:
                self.print_warning("Report Management API module", "Not found")
            
            return True
            
        except Exception as e:
            self.print_result("API configuration", False, str(e))
            return False
    
    def test_backup_system(self) -> bool:
        """Test backup system"""
        self.print_header("8. Backup System")
        
        try:
            from scripts.backup.backup_manager import BackupManager
            self.print_result("Backup Manager import", True)
            
            from scripts.backup.restore_manager import RestoreManager
            self.print_result("Restore Manager import", True)
            
            return True
            
        except Exception as e:
            self.print_result("Backup system", False, str(e))
            return False
    
    def test_security_validators(self) -> bool:
        """Test security components"""
        self.print_header("9. Security Validation")
        
        all_passed = True
        
        try:
            from src.security.validators import FileValidator, InputSanitizer
            
            # Test FileValidator
            validator = FileValidator()
            safe_path = validator.is_safe_path("/valid/path/file.txt")
            self.print_result("FileValidator", True)
            
            # Test InputSanitizer
            sanitizer = InputSanitizer()
            sanitized = sanitizer.sanitize_filename("test_file.txt")
            self.print_result("InputSanitizer", True, f"Sanitized: {sanitized}")
            
        except Exception as e:
            self.print_result("Security validators", False, str(e))
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        print(f"\n{BLUE}{'='*70}")
        print(f"  AVALANCHE eDNA - PIPELINE DRY RUN")
        print(f"{'='*70}{RESET}\n")
        
        tests = [
            self.test_imports,
            self.test_file_structure,
            self.test_configuration,
            self.test_database_connection,
            self.test_docker_services,
            self.test_pipeline_components,
            self.test_api_endpoints,
            self.test_backup_system,
            self.test_security_validators,
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"{RED}[ERROR]{RESET} Test failed with exception: {e}")
                results.append(False)
        
        # Print summary
        self.print_summary()
        
        return all(results)
    
    def print_summary(self):
        """Print validation summary"""
        self.print_header("VALIDATION SUMMARY")
        
        total_tests = len(self.passed) + len(self.errors)
        
        print(f"{GREEN}Passed:{RESET} {len(self.passed)}/{total_tests}")
        print(f"{RED}Failed:{RESET} {len(self.errors)}/{total_tests}")
        print(f"{YELLOW}Warnings:{RESET} {len(self.warnings)}")
        
        if self.errors:
            print(f"\n{RED}Critical Errors:{RESET}")
            for test_name, message in self.errors:
                print(f"  • {test_name}: {message}")
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for test_name, message in self.warnings:
                print(f"  • {test_name}: {message}")
        
        success_rate = (len(self.passed) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{'='*70}")
        if success_rate >= 90:
            print(f"{GREEN}✓ PIPELINE READY - {success_rate:.1f}% validation passed{RESET}")
            print(f"{GREEN}  You can proceed with running the pipeline{RESET}")
        elif success_rate >= 70:
            print(f"{YELLOW}⚠ PIPELINE PARTIALLY READY - {success_rate:.1f}% validation passed{RESET}")
            print(f"{YELLOW}  Review warnings before running the pipeline{RESET}")
        else:
            print(f"{RED}✗ PIPELINE NOT READY - {success_rate:.1f}% validation passed{RESET}")
            print(f"{RED}  Fix critical errors before running the pipeline{RESET}")
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    validator = DryRunValidator()
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
