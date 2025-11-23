#!/usr/bin/env python3
"""
Focused test for Authentication and Analysis Pipeline bugs
"""

import os
import sys

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def test_authentication():
    """Test authentication module imports and functionality"""
    print(f"\n{BLUE}{'='*70}")
    print(f"  TESTING AUTHENTICATION SYSTEM")
    print(f"{'='*70}{RESET}\n")
    
    passed = 0
    failed = 0
    
    # Test 1: Import password utilities
    try:
        sys.path.insert(0, '/app/src/auth')
        from password_utils import validate_password, hash_password, verify_password
        print(f"{GREEN}✅ PASS{RESET} - Password utilities imported successfully")
        passed += 1
    except ImportError as e:
        print(f"{RED}❌ FAIL{RESET} - Cannot import password utilities: {e}")
        failed += 1
        return passed, failed
    
    # Test 2: Password validation
    try:
        is_valid, msg = validate_password("Test@123")
        if is_valid:
            print(f"{GREEN}✅ PASS{RESET} - Password validation working (strong password)")
            passed += 1
        else:
            print(f"{RED}❌ FAIL{RESET} - Strong password rejected: {msg}")
            failed += 1
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET} - Password validation error: {e}")
        failed += 1
    
    # Test 3: Weak password rejection
    try:
        is_valid, msg = validate_password("weak")
        if not is_valid:
            print(f"{GREEN}✅ PASS{RESET} - Weak password correctly rejected: {msg}")
            passed += 1
        else:
            print(f"{RED}❌ FAIL{RESET} - Weak password accepted")
            failed += 1
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET} - Weak password test error: {e}")
        failed += 1
    
    # Test 4: Password hashing
    try:
        test_password = "SecureP@ss123"
        hashed = hash_password(test_password)
        if hashed and len(hashed) > 20:
            print(f"{GREEN}✅ PASS{RESET} - Password hashing working (hash length: {len(hashed)})")
            passed += 1
            
            # Test 5: Password verification
            verified = verify_password(test_password, hashed)
            if verified:
                print(f"{GREEN}✅ PASS{RESET} - Password verification working (correct password)")
                passed += 1
            else:
                print(f"{RED}❌ FAIL{RESET} - Password verification failed for correct password")
                failed += 1
            
            # Test 6: Wrong password rejection
            wrong_verified = verify_password("WrongPassword!", hashed)
            if not wrong_verified:
                print(f"{GREEN}✅ PASS{RESET} - Wrong password correctly rejected")
                passed += 1
            else:
                print(f"{RED}❌ FAIL{RESET} - Wrong password accepted")
                failed += 1
        else:
            print(f"{RED}❌ FAIL{RESET} - Password hashing failed")
            failed += 1
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET} - Password hashing error: {e}")
        failed += 1
    
    # Test 7: PostgresUserManager integration
    try:
        sys.path.insert(0, '/app/src/auth')
        from postgres_user_manager import PostgresUserManager
        manager = PostgresUserManager()
        print(f"{GREEN}✅ PASS{RESET} - PostgresUserManager initialized successfully")
        passed += 1
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET} - PostgresUserManager initialization error: {e}")
        failed += 1
    
    return passed, failed


def test_analysis_pipeline():
    """Test analysis pipeline imports and availability"""
    print(f"\n{BLUE}{'='*70}")
    print(f"  TESTING ANALYSIS PIPELINE")
    print(f"{'='*70}{RESET}\n")
    
    passed = 0
    failed = 0
    
    # Test 1: Import analysis module
    try:
        sys.path.insert(0, '/app')
        from src.analysis import DatasetAnalyzer
        print(f"{GREEN}✅ PASS{RESET} - DatasetAnalyzer imported successfully")
        passed += 1
    except ImportError as e:
        print(f"{RED}❌ FAIL{RESET} - Cannot import DatasetAnalyzer: {e}")
        failed += 1
    
    # Test 2: Import AdvancedTaxonomicAnalyzer
    try:
        from src.analysis import AdvancedTaxonomicAnalyzer
        print(f"{GREEN}✅ PASS{RESET} - AdvancedTaxonomicAnalyzer imported successfully")
        passed += 1
    except ImportError as e:
        print(f"{RED}❌ FAIL{RESET} - Cannot import AdvancedTaxonomicAnalyzer: {e}")
        failed += 1
    
    # Test 3: Import EnhancedDiversityAnalyzer
    try:
        from src.analysis import EnhancedDiversityAnalyzer
        print(f"{GREEN}✅ PASS{RESET} - EnhancedDiversityAnalyzer imported successfully")
        passed += 1
    except ImportError as e:
        print(f"{RED}❌ FAIL{RESET} - Cannot import EnhancedDiversityAnalyzer: {e}")
        failed += 1
    
    # Test 4: Import EnvironmentalContextAnalyzer
    try:
        from src.analysis import EnvironmentalContextAnalyzer
        print(f"{GREEN}✅ PASS{RESET} - EnvironmentalContextAnalyzer imported successfully")
        passed += 1
    except ImportError as e:
        print(f"{RED}❌ FAIL{RESET} - Cannot import EnvironmentalContextAnalyzer: {e}")
        failed += 1
    
    # Test 5: Check results directory
    results_dir = '/app/consolidated_data/results'
    if os.path.exists(results_dir):
        print(f"{GREEN}✅ PASS{RESET} - Results directory exists: {results_dir}")
        passed += 1
    else:
        print(f"{RED}❌ FAIL{RESET} - Results directory missing: {results_dir}")
        failed += 1
    
    # Test 6: Check data directories
    data_dirs = [
        '/app/data/raw',
        '/app/data/processed',
        '/app/data/reference'
    ]
    
    all_exist = True
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            all_exist = False
            print(f"{RED}❌ FAIL{RESET} - Data directory missing: {data_dir}")
            failed += 1
            break
    
    if all_exist:
        print(f"{GREEN}✅ PASS{RESET} - All data directories exist")
        passed += 1
    
    # Test 7: Check BLAST database indices
    indices_dir = '/app/reference/indices'
    if os.path.exists(indices_dir):
        blast_files = [f for f in os.listdir(indices_dir) if f.endswith(('.nhr', '.nin', '.nsq'))]
        if len(blast_files) > 0:
            print(f"{GREEN}✅ PASS{RESET} - BLAST database indices found: {len(blast_files)} files")
            passed += 1
        else:
            print(f"{RED}❌ FAIL{RESET} - No BLAST index files found")
            failed += 1
    else:
        print(f"{RED}❌ FAIL{RESET} - BLAST indices directory missing: {indices_dir}")
        failed += 1
    
    # Test 8: Verify analysis module __all__ exports
    try:
        from src import analysis
        expected_exports = ['DatasetAnalyzer', 'AdvancedTaxonomicAnalyzer', 
                          'EnhancedDiversityAnalyzer', 'EnvironmentalContextAnalyzer']
        
        all_exported = all(export in analysis.__all__ for export in expected_exports)
        if all_exported:
            print(f"{GREEN}✅ PASS{RESET} - All analyzers properly exported in __all__")
            passed += 1
        else:
            missing = [e for e in expected_exports if e not in analysis.__all__]
            print(f"{RED}❌ FAIL{RESET} - Missing exports: {missing}")
            failed += 1
    except Exception as e:
        print(f"{RED}❌ FAIL{RESET} - Error checking exports: {e}")
        failed += 1
    
    return passed, failed


def main():
    """Run all bug verification tests"""
    print(f"\n{BLUE}{'='*70}")
    print(f"  AUTHENTICATION & ANALYSIS PIPELINE BUG FIX VERIFICATION")
    print(f"{'='*70}{RESET}\n")
    
    # Test authentication
    auth_passed, auth_failed = test_authentication()
    
    # Test analysis pipeline
    analysis_passed, analysis_failed = test_analysis_pipeline()
    
    # Summary
    total_passed = auth_passed + analysis_passed
    total_failed = auth_failed + analysis_failed
    total_tests = total_passed + total_failed
    
    print(f"\n{BLUE}{'='*70}")
    print(f"  TEST SUMMARY")
    print(f"{'='*70}{RESET}\n")
    
    print(f"Authentication Tests:    {GREEN}{auth_passed} passed{RESET} / {RED}{auth_failed} failed{RESET}")
    print(f"Analysis Pipeline Tests: {GREEN}{analysis_passed} passed{RESET} / {RED}{analysis_failed} failed{RESET}")
    print(f"\nTotal:                   {GREEN}{total_passed} passed{RESET} / {RED}{total_failed} failed{RESET}")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate:            {success_rate:.1f}%")
    
    if total_failed == 0:
        print(f"\n{GREEN}🎉 ALL BUGS FIXED! All tests passed.{RESET}\n")
        return 0
    else:
        print(f"\n{RED}⚠️  {total_failed} test(s) still failing.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
