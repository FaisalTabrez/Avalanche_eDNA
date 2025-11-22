"""
Application startup script
Initializes all performance optimizations and starts the application
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.cache import cache as redis_cache
from src.utils.fastapi_integration import get_rate_limiter

logger = get_logger(__name__)


def check_redis_connection(max_retries=5, retry_delay=2):
    """
    Check Redis connection with retries
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries
    
    Returns:
        bool: True if connected, False otherwise
    """
    logger.info("Checking Redis connection...")
    
    for attempt in range(1, max_retries + 1):
        try:
            # Test connection
            redis_cache.client.ping()
            logger.info(f"✓ Redis connection successful")
            return True
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    logger.error("✗ Redis connection failed after all retries")
    return False


def init_cache():
    """
    Initialize cache system
    
    Returns:
        bool: True if successful
    """
    logger.info("Initializing cache system...")
    
    try:
        # Get cache configuration
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        logger.info(f"Redis config: {redis_host}:{redis_port}/{redis_db}")
        
        # Check connection
        if not check_redis_connection():
            logger.warning("Cache system initialization failed - continuing without cache")
            return False
        
        # Test cache operations
        test_key = 'startup_test'
        test_value = {'timestamp': time.time(), 'status': 'ok'}
        
        redis_cache.set(test_key, test_value, ttl=60)
        retrieved = redis_cache.get(test_key)
        
        if retrieved == test_value:
            logger.info("✓ Cache operations verified")
            redis_cache.delete(test_key)
        else:
            logger.warning("Cache verification failed")
            return False
        
        logger.info("✓ Cache system initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"✗ Cache initialization error: {e}")
        return False


def init_rate_limiting():
    """
    Initialize rate limiting system
    
    Returns:
        bool: True if successful
    """
    logger.info("Initializing rate limiting system...")
    
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        logger.info(f"Rate limiting backend: {redis_url}")
        
        # Initialize rate limiter
        limiter = get_rate_limiter(redis_url=redis_url)
        
        logger.info("✓ Rate limiting system initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"✗ Rate limiting initialization error: {e}")
        return False


def init_database():
    """
    Initialize database optimizations
    
    Returns:
        bool: True if successful
    """
    logger.info("Initializing database optimizations...")
    
    try:
        # Import here to avoid circular dependencies
        from scripts.init_database import init_database_optimizations
        
        # Get environment
        env = os.getenv('ENVIRONMENT', 'production')
        
        # Initialize database
        success = init_database_optimizations(
            environment=env,
            create_idx=True,
            run_vacuum=False,  # Don't run VACUUM on startup
            show_stats=False   # Don't show stats on startup
        )
        
        if success:
            logger.info("✓ Database optimizations initialized successfully")
        else:
            logger.warning("Database optimization initialization had issues")
        
        return success
    
    except Exception as e:
        logger.error(f"✗ Database initialization error: {e}")
        logger.warning("Continuing without database optimizations...")
        return False


def init_monitoring():
    """
    Initialize monitoring and metrics
    
    Returns:
        bool: True if successful
    """
    logger.info("Initializing monitoring system...")
    
    try:
        # Import monitoring components
        from src.monitoring.metrics import init_metrics
        
        # Initialize metrics
        init_metrics()
        
        logger.info("✓ Monitoring system initialized successfully")
        return True
    
    except Exception as e:
        logger.warning(f"Monitoring initialization skipped: {e}")
        return False


def startup_checks():
    """
    Run comprehensive startup checks
    
    Returns:
        dict: Status of each component
    """
    logger.info("=" * 80)
    logger.info("AVALANCHE eDNA - APPLICATION STARTUP")
    logger.info("=" * 80)
    
    results = {}
    
    # 1. Cache
    logger.info("\n[1/4] Cache System")
    logger.info("-" * 80)
    results['cache'] = init_cache()
    
    # 2. Rate Limiting
    logger.info("\n[2/4] Rate Limiting")
    logger.info("-" * 80)
    results['rate_limiting'] = init_rate_limiting()
    
    # 3. Database
    logger.info("\n[3/4] Database Optimizations")
    logger.info("-" * 80)
    results['database'] = init_database()
    
    # 4. Monitoring
    logger.info("\n[4/4] Monitoring & Metrics")
    logger.info("-" * 80)
    results['monitoring'] = init_monitoring()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STARTUP SUMMARY")
    logger.info("=" * 80)
    
    for component, status in results.items():
        status_str = "✓ OK" if status else "✗ FAILED"
        logger.info(f"{component.upper():<20} {status_str}")
    
    # Overall status
    critical_components = ['cache', 'rate_limiting']
    all_critical_ok = all(results.get(comp, False) for comp in critical_components)
    
    if all_critical_ok:
        logger.info("\n✓ All critical components initialized successfully")
        logger.info("=" * 80)
        return True
    else:
        logger.warning("\n⚠ Some components failed to initialize")
        logger.warning("Application may run with reduced functionality")
        logger.info("=" * 80)
        return False


def main():
    """
    Main startup function
    """
    success = startup_checks()
    
    if not success:
        logger.warning("Starting application despite initialization warnings...")
    
    # Start application based on environment
    app_type = os.getenv('APP_TYPE', 'streamlit')
    
    if app_type == 'api':
        logger.info("\nStarting FastAPI server...")
        import uvicorn
        from src.api.report_management_api import app
        
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', '8000'))
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level='info',
            access_log=True
        )
    
    elif app_type == 'streamlit':
        logger.info("\nStreamlit app should be started via streamlit command")
        logger.info("Run: streamlit run streamlit_app.py")
    
    else:
        logger.info("\nNo application type specified")
        logger.info("Set APP_TYPE=api or APP_TYPE=streamlit")


if __name__ == '__main__':
    main()
