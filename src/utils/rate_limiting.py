"""
Rate limiting and throttling utilities
Implements token bucket and sliding window algorithms
"""
import time
import functools
from typing import Optional, Callable, Tuple
from flask import request, jsonify, g
import logging

from src.utils.cache import cache

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter using Redis"""
    
    def __init__(
        self,
        cache_client=None,
        default_limit: int = 100,
        default_window: int = 60
    ):
        """
        Initialize rate limiter
        
        Args:
            cache_client: Cache client (uses global cache if None)
            default_limit: Default number of requests allowed
            default_window: Default time window in seconds
        """
        self.cache = cache_client or cache
        self.default_limit = default_limit
        self.default_window = default_window
    
    def is_allowed(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None
    ) -> Tuple[bool, dict]:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Unique identifier for rate limit (e.g., user_id, IP)
            limit: Number of requests allowed
            window: Time window in seconds
            
        Returns:
            (is_allowed, info_dict)
        """
        limit = limit or self.default_limit
        window = window or self.default_window
        
        # Redis key for this rate limit
        redis_key = f"rate_limit:{key}"
        
        # Get current count
        current = self.cache.get(redis_key)
        
        if current is None:
            # First request in window
            self.cache.set(redis_key, 1, ttl=window)
            return True, {
                'limit': limit,
                'remaining': limit - 1,
                'reset': int(time.time()) + window
            }
        
        current = int(current)
        
        if current < limit:
            # Still under limit
            self.cache.increment(redis_key)
            ttl = self.cache.ttl(redis_key)
            return True, {
                'limit': limit,
                'remaining': limit - current - 1,
                'reset': int(time.time()) + ttl
            }
        
        # Over limit
        ttl = self.cache.ttl(redis_key)
        return False, {
            'limit': limit,
            'remaining': 0,
            'reset': int(time.time()) + ttl,
            'retry_after': ttl
        }
    
    def reset(self, key: str):
        """Reset rate limit for key"""
        redis_key = f"rate_limit:{key}"
        self.cache.delete(redis_key)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more precise limiting"""
    
    def __init__(
        self,
        cache_client=None,
        default_limit: int = 100,
        default_window: int = 60
    ):
        """
        Initialize sliding window rate limiter
        
        Args:
            cache_client: Cache client
            default_limit: Default number of requests allowed
            default_window: Default time window in seconds
        """
        self.cache = cache_client or cache
        self.default_limit = default_limit
        self.default_window = default_window
    
    def is_allowed(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None
    ) -> Tuple[bool, dict]:
        """
        Check if request is allowed using sliding window
        
        Args:
            key: Unique identifier
            limit: Number of requests allowed
            window: Time window in seconds
            
        Returns:
            (is_allowed, info_dict)
        """
        limit = limit or self.default_limit
        window = window or self.default_window
        
        now = time.time()
        redis_key = f"rate_limit:sliding:{key}"
        
        # Get request timestamps
        timestamps = self.cache.get(redis_key) or []
        
        # Remove old timestamps outside window
        cutoff = now - window
        timestamps = [ts for ts in timestamps if ts > cutoff]
        
        # Check if under limit
        if len(timestamps) < limit:
            # Add current timestamp
            timestamps.append(now)
            self.cache.set(redis_key, timestamps, ttl=window)
            
            return True, {
                'limit': limit,
                'remaining': limit - len(timestamps),
                'reset': int(timestamps[0] + window) if timestamps else int(now + window)
            }
        
        # Over limit
        return False, {
            'limit': limit,
            'remaining': 0,
            'reset': int(timestamps[0] + window),
            'retry_after': int(timestamps[0] + window - now)
        }


# Global rate limiters
rate_limiter = RateLimiter()
sliding_rate_limiter = SlidingWindowRateLimiter()


# ============================================================================
# Rate Limit Decorators
# ============================================================================

def rate_limit(
    limit: int = 100,
    window: int = 60,
    key_func: Optional[Callable] = None,
    error_message: str = "Rate limit exceeded"
):
    """
    Decorator to apply rate limiting to Flask routes
    
    Args:
        limit: Number of requests allowed
        window: Time window in seconds
        key_func: Function to generate rate limit key (default: IP address)
        error_message: Error message to return
        
    Example:
        @app.route('/api/search')
        @rate_limit(limit=10, window=60)
        def search():
            return jsonify(results)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                key = key_func()
            else:
                # Default: use IP address
                key = request.remote_addr or 'unknown'
            
            # Check rate limit
            allowed, info = rate_limiter.is_allowed(key, limit, window)
            
            # Add rate limit headers
            g.rate_limit_info = info
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {key}")
                response = jsonify({
                    'error': error_message,
                    'limit': info['limit'],
                    'reset': info['reset'],
                    'retry_after': info.get('retry_after')
                })
                response.status_code = 429
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = str(info['reset'])
                response.headers['Retry-After'] = str(info.get('retry_after', window))
                return response
            
            # Execute function
            response = func(*args, **kwargs)
            
            # Add rate limit headers to successful response
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset'])
            
            return response
        
        return wrapper
    return decorator


def throttle(
    calls: int = 1,
    period: int = 1,
    key_func: Optional[Callable] = None
):
    """
    Decorator to throttle function calls
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
        key_func: Function to generate throttle key
        
    Example:
        @throttle(calls=5, period=60)
        def expensive_operation():
            # Only allows 5 calls per minute
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate throttle key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}"
            
            # Check throttle
            allowed, info = rate_limiter.is_allowed(key, calls, period)
            
            if not allowed:
                wait_time = info.get('retry_after', period)
                logger.warning(
                    f"Throttle limit exceeded for {key}. "
                    f"Retry after {wait_time}s"
                )
                raise TooManyRequestsError(
                    f"Too many requests. Retry after {wait_time} seconds",
                    retry_after=wait_time
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# Per-User Rate Limiting
# ============================================================================

def user_rate_limit(
    limit: int = 1000,
    window: int = 3600,
    error_message: str = "User rate limit exceeded"
):
    """
    Rate limit per authenticated user
    
    Args:
        limit: Number of requests allowed per user
        window: Time window in seconds
        error_message: Error message
        
    Example:
        @app.route('/api/datasets')
        @login_required
        @user_rate_limit(limit=100, window=3600)
        def list_datasets():
            return jsonify(datasets)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get user ID from request context
            user_id = getattr(g, 'user_id', None)
            
            if not user_id:
                # No user context, skip rate limiting
                return func(*args, **kwargs)
            
            # Check rate limit for user
            key = f"user:{user_id}"
            allowed, info = rate_limiter.is_allowed(key, limit, window)
            
            if not allowed:
                logger.warning(f"User rate limit exceeded for user {user_id}")
                response = jsonify({
                    'error': error_message,
                    'limit': info['limit'],
                    'reset': info['reset']
                })
                response.status_code = 429
                return response
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# Endpoint-Specific Rate Limiting
# ============================================================================

# Rate limit configurations for different endpoints
ENDPOINT_LIMITS = {
    # Authentication endpoints - strict limits
    '/api/auth/login': {'limit': 5, 'window': 300},  # 5 per 5 min
    '/api/auth/register': {'limit': 3, 'window': 3600},  # 3 per hour
    '/api/auth/reset-password': {'limit': 3, 'window': 3600},
    
    # Search/query endpoints - moderate limits
    '/api/search': {'limit': 20, 'window': 60},  # 20 per minute
    '/api/datasets/search': {'limit': 30, 'window': 60},
    
    # Upload endpoints - low limits
    '/api/datasets/upload': {'limit': 10, 'window': 3600},  # 10 per hour
    '/api/reports/create': {'limit': 20, 'window': 3600},
    
    # Read endpoints - high limits
    '/api/datasets': {'limit': 100, 'window': 60},
    '/api/reports': {'limit': 100, 'window': 60},
    '/api/analysis/results': {'limit': 50, 'window': 60},
    
    # Heavy computation endpoints - very low limits
    '/api/analysis/start': {'limit': 5, 'window': 300},  # 5 per 5 min
    '/api/training/start': {'limit': 3, 'window': 3600},  # 3 per hour
    '/api/download/sra': {'limit': 10, 'window': 3600},
}


def get_endpoint_limit(path: str) -> dict:
    """
    Get rate limit configuration for endpoint
    
    Args:
        path: Request path
        
    Returns:
        Dict with 'limit' and 'window' keys
    """
    # Check exact match
    if path in ENDPOINT_LIMITS:
        return ENDPOINT_LIMITS[path]
    
    # Check prefix match
    for endpoint_path, limits in ENDPOINT_LIMITS.items():
        if path.startswith(endpoint_path):
            return limits
    
    # Default limits
    return {'limit': 100, 'window': 60}


# ============================================================================
# Exceptions
# ============================================================================

class TooManyRequestsError(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


# ============================================================================
# Flask Integration
# ============================================================================

def init_rate_limiting(app):
    """
    Initialize rate limiting for Flask app
    
    Args:
        app: Flask application
    """
    @app.after_request
    def add_rate_limit_headers(response):
        """Add rate limit headers to all responses"""
        if hasattr(g, 'rate_limit_info'):
            info = g.rate_limit_info
            response.headers['X-RateLimit-Limit'] = str(info.get('limit', ''))
            response.headers['X-RateLimit-Remaining'] = str(info.get('remaining', ''))
            response.headers['X-RateLimit-Reset'] = str(info.get('reset', ''))
        
        return response
    
    @app.errorhandler(429)
    def handle_rate_limit_error(error):
        """Handle rate limit errors"""
        response = jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.'
        })
        response.status_code = 429
        return response
    
    logger.info("Rate limiting initialized")
