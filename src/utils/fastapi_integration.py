"""
FastAPI integration for caching and rate limiting
Adapts Flask-based decorators for FastAPI compatibility
"""
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from functools import wraps
from typing import Callable, Optional
import hashlib
import json
import time

from src.utils.cache import CacheClient, cache as default_cache
from src.utils.rate_limiting import RateLimiter, SlidingWindowRateLimiter, TooManyRequestsError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# FastAPI Cache Decorators
# ============================================================================

def fastapi_cached(
    ttl: int = 3600,
    key_prefix: str = '',
    cache_client: Optional[CacheClient] = None
):
    """
    Cache decorator for FastAPI endpoints
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        cache_client: Custom cache client (uses default if None)
    
    Returns:
        Decorator function
    
    Example:
        @app.get("/api/datasets")
        @fastapi_cached(ttl=600, key_prefix='datasets')
        async def list_datasets():
            return get_all_datasets()
    """
    cache = cache_client or default_cache
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs if present
            request = kwargs.get('request')
            
            # Generate cache key
            if request:
                cache_key = _generate_request_cache_key(request, key_prefix)
            else:
                # Fallback to function args
                cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs) if callable(getattr(func, '__wrapped__', None)) and hasattr(func.__wrapped__, '__await__') else func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    
    return decorator


def _generate_request_cache_key(request: Request, prefix: str = '') -> str:
    """
    Generate cache key from FastAPI request
    
    Args:
        request: FastAPI Request object
        prefix: Key prefix
    
    Returns:
        Cache key string
    """
    # Include method, path, and query params in key
    method = request.method
    path = request.url.path
    query_params = str(sorted(request.query_params.items()))
    
    key_base = f"{prefix}:{method}:{path}:{query_params}"
    
    # Hash if too long
    if len(key_base) > 200:
        key_hash = hashlib.md5(key_base.encode()).hexdigest()
        return f"{prefix}:hashed:{key_hash}"
    
    return key_base


# ============================================================================
# FastAPI Rate Limiting
# ============================================================================

class FastAPIRateLimiter:
    """
    Rate limiter for FastAPI applications
    """
    
    def __init__(self, redis_url: str = 'redis://localhost:6379/0'):
        """
        Initialize rate limiter
        
        Args:
            redis_url: Redis connection URL
        """
        self.limiters = {}  # Cache of RateLimiter instances
        self.redis_url = redis_url
    
    def get_limiter(self, limit: int, window: int) -> RateLimiter:
        """
        Get or create rate limiter for given limit/window
        
        Args:
            limit: Number of requests allowed
            window: Time window in seconds
        
        Returns:
            RateLimiter instance
        """
        key = f"{limit}:{window}"
        if key not in self.limiters:
            self.limiters[key] = RateLimiter(
                redis_url=self.redis_url,
                max_requests=limit,
                window_seconds=window
            )
        return self.limiters[key]
    
    def check_rate_limit(
        self,
        request: Request,
        limit: int,
        window: int,
        key_func: Optional[Callable] = None
    ) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            request: FastAPI Request object
            limit: Number of requests allowed
            window: Time window in seconds
            key_func: Function to generate rate limit key (default: IP address)
        
        Returns:
            True if allowed, raises HTTPException if exceeded
        """
        # Generate rate limit key
        if key_func:
            rate_key = key_func(request)
        else:
            # Default: use client IP
            rate_key = request.client.host if request.client else 'unknown'
        
        # Get limiter
        limiter = self.get_limiter(limit, window)
        
        # Check limit
        try:
            allowed, retry_after = limiter.allow(rate_key)
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        'error': 'Too Many Requests',
                        'message': f'Rate limit exceeded. Try again in {retry_after} seconds.',
                        'retry_after': retry_after
                    },
                    headers={
                        'X-RateLimit-Limit': str(limit),
                        'X-RateLimit-Window': str(window),
                        'Retry-After': str(int(retry_after))
                    }
                )
            
            return True
        
        except TooManyRequestsError as e:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'Too Many Requests',
                    'message': str(e)
                }
            )


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter(redis_url: str = 'redis://localhost:6379/0') -> FastAPIRateLimiter:
    """
    Get global rate limiter instance
    
    Args:
        redis_url: Redis connection URL
    
    Returns:
        FastAPIRateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = FastAPIRateLimiter(redis_url=redis_url)
    return _rate_limiter


def fastapi_rate_limit(
    limit: int,
    window: int,
    key_func: Optional[Callable] = None
):
    """
    Rate limiting decorator for FastAPI endpoints
    
    Args:
        limit: Number of requests allowed
        window: Time window in seconds
        key_func: Function to generate rate limit key
    
    Returns:
        Decorator function
    
    Example:
        @app.get("/api/search")
        @fastapi_rate_limit(limit=30, window=60)
        async def search(request: Request):
            return perform_search()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs
            request = kwargs.get('request')
            
            if request:
                # Get rate limiter
                limiter = get_rate_limiter()
                
                # Check rate limit
                limiter.check_rate_limit(request, limit, window, key_func)
            
            # Execute function
            return await func(*args, **kwargs) if callable(getattr(func, '__wrapped__', None)) and hasattr(func.__wrapped__, '__await__') else func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# FastAPI Middleware
# ============================================================================

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware for caching FastAPI responses
    """
    
    def __init__(self, app, cache_client: Optional[CacheClient] = None, default_ttl: int = 3600):
        """
        Initialize cache middleware
        
        Args:
            app: FastAPI application
            cache_client: Cache client instance
            default_ttl: Default time-to-live in seconds
        """
        super().__init__(app)
        self.cache = cache_client or default_cache
        self.default_ttl = default_ttl
    
    async def dispatch(self, request: StarletteRequest, call_next):
        """
        Process request with caching
        
        Args:
            request: Starlette Request
            call_next: Next middleware in chain
        
        Returns:
            Response
        """
        # Only cache GET requests
        if request.method != 'GET':
            return await call_next(request)
        
        # Generate cache key
        cache_key = _generate_request_cache_key(request, prefix='middleware')
        
        # Try cache
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            logger.debug(f"Middleware cache hit: {cache_key}")
            return JSONResponse(content=cached_response)
        
        # Execute request
        logger.debug(f"Middleware cache miss: {cache_key}")
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            try:
                # Try to parse as JSON
                content = json.loads(body.decode())
                self.cache.set(cache_key, content, ttl=self.default_ttl)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not JSON, skip caching
                pass
            
            # Return response with body
            return StarletteResponse(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting FastAPI requests
    """
    
    def __init__(
        self,
        app,
        default_limit: int = 100,
        default_window: int = 60,
        redis_url: str = 'redis://localhost:6379/0'
    ):
        """
        Initialize rate limit middleware
        
        Args:
            app: FastAPI application
            default_limit: Default request limit
            default_window: Default time window in seconds
            redis_url: Redis connection URL
        """
        super().__init__(app)
        self.default_limit = default_limit
        self.default_window = default_window
        self.limiter = FastAPIRateLimiter(redis_url=redis_url)
    
    async def dispatch(self, request: StarletteRequest, call_next):
        """
        Process request with rate limiting
        
        Args:
            request: Starlette Request
            call_next: Next middleware in chain
        
        Returns:
            Response
        """
        # Check rate limit
        try:
            self.limiter.check_rate_limit(
                request,
                self.default_limit,
                self.default_window
            )
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail,
                headers=e.headers or {}
            )
        
        # Continue processing
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers['X-RateLimit-Limit'] = str(self.default_limit)
        response.headers['X-RateLimit-Window'] = str(self.default_window)
        
        return response


# ============================================================================
# Initialization Helper
# ============================================================================

def init_fastapi_optimizations(
    app,
    enable_cache: bool = True,
    enable_rate_limit: bool = True,
    cache_ttl: int = 3600,
    rate_limit: int = 100,
    rate_window: int = 60,
    redis_url: str = 'redis://localhost:6379/0'
):
    """
    Initialize caching and rate limiting for FastAPI app
    
    Args:
        app: FastAPI application
        enable_cache: Enable caching middleware
        enable_rate_limit: Enable rate limiting middleware
        cache_ttl: Default cache TTL in seconds
        rate_limit: Default rate limit (requests)
        rate_window: Default rate window (seconds)
        redis_url: Redis connection URL
    
    Example:
        from fastapi import FastAPI
        from src.utils.fastapi_integration import init_fastapi_optimizations
        
        app = FastAPI()
        init_fastapi_optimizations(
            app,
            cache_ttl=600,
            rate_limit=50,
            rate_window=60
        )
    """
    if enable_cache:
        app.add_middleware(
            CacheMiddleware,
            cache_client=default_cache,
            default_ttl=cache_ttl
        )
        logger.info(f"Cache middleware enabled (TTL: {cache_ttl}s)")
    
    if enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=rate_limit,
            default_window=rate_window,
            redis_url=redis_url
        )
        logger.info(f"Rate limit middleware enabled ({rate_limit} req/{rate_window}s)")
