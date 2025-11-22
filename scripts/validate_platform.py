#!/usr/bin/env python
"""
Quick validation test for the Avalanche eDNA platform
Tests all critical components and services
"""
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from colorama import init, Fore, Style

init(autoreset=True)

def test_section(name):
    """Print test section header"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}  {name}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")

def test_passed(message):
    """Print success message"""
    print(f"{Fore.GREEN}[PASS]{Style.RESET_ALL} {message}")

def test_failed(message, error=None):
    """Print failure message"""
    print(f"{Fore.RED}[FAIL]{Style.RESET_ALL} {message}")
    if error:
        print(f"{Fore.YELLOW}  Error: {error}{Style.RESET_ALL}")
    return False

def main():
    """Run validation tests"""
    print(f"{Fore.MAGENTA}")
    print("="*70)
    print("  AVALANCHE eDNA - PLATFORM VALIDATION")
    print("="*70)
    print(f"{Style.RESET_ALL}")
    
    all_passed = True
    
    # ========================================================================
    # 1. Docker Services
    # ========================================================================
    test_section("1. Docker Services")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        version = r.info('server')['redis_version']
        test_passed(f"Redis connection (v{version})")
    except Exception as e:
        all_passed = test_failed("Redis connection", e)
    
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='avalanche_edna',
            user='avalanche',
            password='avalanche_dev_password'
        )
        conn.close()
        test_passed("PostgreSQL connection")
    except Exception as e:
        all_passed = test_failed("PostgreSQL connection", e)
    
    try:
        import requests
        resp = requests.get('http://localhost:9090/-/healthy', timeout=5)
        if resp.status_code == 200:
            test_passed("Prometheus service")
        else:
            all_passed = test_failed("Prometheus service", f"Status: {resp.status_code}")
    except Exception as e:
        all_passed = test_failed("Prometheus service", e)
    
    try:
        resp = requests.get('http://localhost:3000/api/health', timeout=5)
        if resp.status_code == 200:
            test_passed("Grafana service")
        else:
            all_passed = test_failed("Grafana service", f"Status: {resp.status_code}")
    except Exception as e:
        all_passed = test_failed("Grafana service", e)
    
    # ========================================================================
    # 2. Phase 3 Optimizations
    # ========================================================================
    test_section("2. Phase 3 Optimizations")
    
    try:
        from src.utils.cache import CacheClient
        cache = CacheClient()
        cache.set("validation_test", "ok", ttl=10)
        value = cache.get("validation_test")
        if value == "ok":
            test_passed("Cache operations (set/get)")
        else:
            all_passed = test_failed("Cache operations", f"Expected 'ok', got '{value}'")
    except Exception as e:
        all_passed = test_failed("Cache operations", e)
    
    try:
        from src.utils.rate_limiting import RateLimiter
        limiter = RateLimiter(default_limit=10, default_window=60)
        allowed, info = limiter.is_allowed("validation_user", limit=5, window=60)
        if allowed and info['remaining'] >= 0:
            test_passed(f"Rate limiting (remaining: {info['remaining']})")
        else:
            all_passed = test_failed("Rate limiting", "Request denied unexpectedly")
    except Exception as e:
        all_passed = test_failed("Rate limiting", e)
    
    try:
        from src.utils.fastapi_integration import get_rate_limiter, fastapi_cached
        limiter = get_rate_limiter()
        if limiter is not None:
            test_passed("FastAPI integration")
        else:
            all_passed = test_failed("FastAPI integration", "Rate limiter is None")
    except Exception as e:
        all_passed = test_failed("FastAPI integration", e)
    
    # ========================================================================
    # 3. Core Components
    # ========================================================================
    test_section("3. Core Components")
    
    try:
        from src.models.tokenizer import DNATokenizer
        tokenizer = DNATokenizer(kmer_size=3)
        seq = "ATCGATCG"
        encoded = tokenizer.encode_sequence(seq)
        if encoded is not None:
            test_passed("DNA Tokenizer")
        else:
            all_passed = test_failed("DNA Tokenizer", "Encoding returned None")
    except Exception as e:
        all_passed = test_failed("DNA Tokenizer", e)
    
    try:
        from src.clustering.multi_source_taxonomy import MultiSourceTaxonomyResolver
        resolver = MultiSourceTaxonomyResolver()
        test_passed("Taxonomy Resolver")
    except Exception as e:
        all_passed = test_failed("Taxonomy Resolver", e)
    
    try:
        from src.database import DatabaseConnection, DatabaseConfig
        config = DatabaseConfig()
        test_passed("Database Configuration")
    except Exception as e:
        all_passed = test_failed("Database Configuration", e)
    
    # ========================================================================
    # 4. API Components
    # ========================================================================
    test_section("4. API Components")
    
    try:
        from src.api.report_management_api import app
        test_passed("Report Management API")
    except Exception as e:
        all_passed = test_failed("Report Management API", e)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    if all_passed:
        print(f"{Fore.GREEN}")
        print("  [PASS] ALL VALIDATION TESTS PASSED")
        print("  Platform is ready for use!")
        print(f"{Style.RESET_ALL}")
        return 0
    else:
        print(f"{Fore.YELLOW}")
        print("  [WARN] SOME VALIDATION TESTS FAILED")
        print("  Please check the errors above")
        print(f"{Style.RESET_ALL}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
