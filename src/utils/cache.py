"""
Caching utilities for performance optimization
Provides Redis-based caching, cache decorators, and invalidation strategies
"""
import functools
import hashlib
import json
import pickle
from typing import Any, Callable, Optional, Union
from datetime import timedelta
import redis
from flask import request, make_response
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Redis Cache Client
# ============================================================================

class CacheClient:
    """Redis cache client with connection pooling"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = False
    ):
        """
        Initialize Redis cache client
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            decode_responses: Whether to decode responses to strings
        """
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=decode_responses
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        try:
            value = self.client.get(key)
            if value is None:
                return default
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = default_ttl)
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        try:
            ttl = ttl if ttl is not None else self.default_ttl
            serialized = pickle.dumps(value)
            return self.client.set(key, serialized, ex=ttl, nx=nx, xx=xx)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, *keys: str) -> int:
        """
        Delete keys from cache
        
        Args:
            keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            if not keys:
                return 0
            return self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return 0
    
    def exists(self, *keys: str) -> int:
        """
        Check if keys exist
        
        Args:
            keys: Keys to check
            
        Returns:
            Number of keys that exist
        """
        try:
            return self.client.exists(*keys)
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return 0
    
    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration on key
        
        Args:
            key: Cache key
            seconds: Seconds until expiration
            
        Returns:
            True if successful
        """
        try:
            return self.client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        Get time to live for key
        
        Args:
            key: Cache key
            
        Returns:
            Seconds until expiration (-1 if no expiry, -2 if key doesn't exist)
        """
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -2
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment value
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value or None on error
        """
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Decrement value
        
        Args:
            key: Cache key
            amount: Amount to decrement
            
        Returns:
            New value or None on error
        """
        try:
            return self.client.decrby(key, amount)
        except Exception as e:
            logger.error(f"Cache decrement error for key {key}: {e}")
            return None
    
    def flush(self) -> bool:
        """
        Flush all keys in current database
        
        Returns:
            True if successful
        """
        try:
            return self.client.flushdb()
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False
    
    def keys(self, pattern: str = '*') -> list:
        """
        Get keys matching pattern
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            List of matching keys
        """
        try:
            return [k.decode() if isinstance(k, bytes) else k 
                    for k in self.client.keys(pattern)]
        except Exception as e:
            logger.error(f"Cache keys error: {e}")
            return []
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Key pattern
            
        Returns:
            Number of keys deleted
        """
        keys = self.keys(pattern)
        if keys:
            return self.delete(*keys)
        return 0


# Global cache instance
cache = CacheClient()


# ============================================================================
# Cache Decorators
# ============================================================================

def cached(
    ttl: Optional[int] = None,
    key_prefix: str = '',
    key_func: Optional[Callable] = None
):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        key_func: Function to generate cache key from args
        
    Example:
        @cached(ttl=300, key_prefix='user')
        def get_user(user_id):
            return db.query(User).get(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs, key_prefix)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value
            
            # Cache miss - execute function
            logger.debug(f"Cache miss for key: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        # Add cache invalidation method
        wrapper.invalidate = lambda *args, **kwargs: _invalidate_cache(
            func, args, kwargs, key_prefix, key_func
        )
        
        return wrapper
    return decorator


def cache_response(
    ttl: int = 300,
    key_prefix: str = 'response',
    vary_on: Optional[list] = None
):
    """
    Decorator to cache Flask response
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        vary_on: List of request attributes to vary cache on
                 (e.g., ['args', 'headers.Accept-Language'])
    
    Example:
        @app.route('/api/datasets')
        @cache_response(ttl=600, vary_on=['args'])
        def list_datasets():
            return jsonify(datasets)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from request
            cache_key = _generate_request_cache_key(
                request, key_prefix, vary_on
            )
            
            # Try to get from cache
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Response cache hit: {cache_key}")
                response = make_response(cached_response['body'])
                response.headers.update(cached_response['headers'])
                response.headers['X-Cache'] = 'HIT'
                return response
            
            # Cache miss - execute function
            logger.debug(f"Response cache miss: {cache_key}")
            response = func(*args, **kwargs)
            
            # Cache response
            if response.status_code == 200:
                cached_data = {
                    'body': response.get_data(as_text=True),
                    'headers': dict(response.headers)
                }
                cache.set(cache_key, cached_data, ttl=ttl)
                response.headers['X-Cache'] = 'MISS'
            
            return response
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str):
    """
    Decorator to invalidate cache after function execution
    
    Args:
        pattern: Cache key pattern to invalidate
        
    Example:
        @invalidate_cache('user:*')
        def update_user(user_id, data):
            # Update user...
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            cache.clear_pattern(pattern)
            logger.info(f"Invalidated cache pattern: {pattern}")
            return result
        return wrapper
    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    prefix: str = ''
) -> str:
    """Generate cache key from function and arguments"""
    # Build key parts
    parts = [prefix] if prefix else []
    parts.append(f"{func.__module__}.{func.__name__}")
    
    # Add arguments
    if args:
        parts.append(str(args))
    if kwargs:
        parts.append(str(sorted(kwargs.items())))
    
    # Hash if key is too long
    key = ':'.join(parts)
    if len(key) > 200:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{prefix}:{func.__name__}:{key_hash}"
    
    return key


def _generate_request_cache_key(
    request_obj,
    prefix: str = 'response',
    vary_on: Optional[list] = None
) -> str:
    """Generate cache key from Flask request"""
    parts = [prefix, request_obj.path]
    
    if vary_on:
        for attribute in vary_on:
            if '.' in attribute:
                # Handle nested attributes like 'headers.Accept-Language'
                obj = request_obj
                for attr in attribute.split('.'):
                    obj = getattr(obj, attr, {})
                    if isinstance(obj, dict):
                        obj = obj.get(attr, '')
                parts.append(str(obj))
            else:
                value = getattr(request_obj, attribute, None)
                if value:
                    parts.append(str(value))
    else:
        # Default: vary on query args
        if request_obj.args:
            parts.append(str(sorted(request_obj.args.items())))
    
    key = ':'.join(parts)
    if len(key) > 200:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    return key


def _invalidate_cache(
    func: Callable,
    args: tuple,
    kwargs: dict,
    prefix: str,
    key_func: Optional[Callable]
):
    """Invalidate cache for specific function call"""
    if key_func:
        cache_key = key_func(*args, **kwargs)
    else:
        cache_key = _generate_cache_key(func, args, kwargs, prefix)
    
    cache.delete(cache_key)
    logger.info(f"Invalidated cache key: {cache_key}")


# ============================================================================
# Cache Warming
# ============================================================================

class CacheWarmer:
    """Utility to pre-populate cache with frequently accessed data"""
    
    def __init__(self, cache_client: CacheClient):
        self.cache = cache_client
    
    def warm(self, key: str, value: Any, ttl: Optional[int] = None):
        """Warm cache with specific value"""
        self.cache.set(key, value, ttl=ttl)
        logger.info(f"Warmed cache key: {key}")
    
    def warm_batch(self, items: dict, ttl: Optional[int] = None):
        """Warm cache with multiple items"""
        for key, value in items.items():
            self.cache.set(key, value, ttl=ttl)
        logger.info(f"Warmed {len(items)} cache keys")
    
    def warm_from_query(
        self,
        query_func: Callable,
        key_func: Callable,
        ttl: Optional[int] = None
    ):
        """
        Warm cache from database query
        
        Args:
            query_func: Function that returns query results
            key_func: Function to generate cache key from each item
            ttl: Time to live
        """
        items = query_func()
        count = 0
        
        for item in items:
            key = key_func(item)
            self.cache.set(key, item, ttl=ttl)
            count += 1
        
        logger.info(f"Warmed {count} items from query")
        return count


# ============================================================================
# Cache Statistics
# ============================================================================

class CacheStats:
    """Track cache hit/miss statistics"""
    
    def __init__(self, cache_client: CacheClient):
        self.cache = cache_client
        self.hits_key = 'cache:stats:hits'
        self.misses_key = 'cache:stats:misses'
    
    def record_hit(self):
        """Record cache hit"""
        self.cache.increment(self.hits_key)
    
    def record_miss(self):
        """Record cache miss"""
        self.cache.increment(self.misses_key)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        hits = int(self.cache.get(self.hits_key) or 0)
        misses = int(self.cache.get(self.misses_key) or 0)
        total = hits + misses
        
        return {
            'hits': hits,
            'misses': misses,
            'total': total,
            'hit_rate': (hits / total * 100) if total > 0 else 0.0
        }
    
    def reset(self):
        """Reset statistics"""
        self.cache.delete(self.hits_key, self.misses_key)


# Initialize cache warmer and stats
cache_warmer = CacheWarmer(cache)
cache_stats = CacheStats(cache)
