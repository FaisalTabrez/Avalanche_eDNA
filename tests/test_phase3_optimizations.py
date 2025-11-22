"""
Tests for Phase 3 performance optimizations
"""
import pytest
import time
from fastapi import Request
from fastapi.testclient import TestClient

from src.utils.cache import CacheClient
from src.utils.rate_limiting import RateLimiter, SlidingWindowRateLimiter
from src.utils.fastapi_integration import (
    CacheMiddleware,
    RateLimitMiddleware,
    get_rate_limiter,
    fastapi_cached,
    fastapi_rate_limit
)


class TestCacheClient:
    """Test cache functionality"""
    
    @pytest.fixture
    def cache_client(self):
        """Create cache client"""
        return CacheClient()
    
    def test_cache_set_get(self, cache_client):
        """Test basic cache operations"""
        cache_client.set("test_key", "test_value", ttl=60)
        result = cache_client.get("test_key")
        assert result == "test_value"
    
    def test_cache_delete(self, cache_client):
        """Test cache deletion"""
        cache_client.set("test_key", "test_value")
        cache_client.delete("test_key")
        result = cache_client.get("test_key")
        assert result is None
    
    def test_cache_expiration(self, cache_client):
        """Test cache TTL"""
        cache_client.set("test_key", "test_value", ttl=1)
        time.sleep(2)
        result = cache_client.get("test_key")
        assert result is None
    
    def test_cache_flush(self, cache_client):
        """Test cache flush"""
        cache_client.set("key1", "value1")
        cache_client.set("key2", "value2")
        cache_client.flush()
        assert cache_client.get("key1") is None
        assert cache_client.get("key2") is None


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_allow(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(default_limit=10, default_window=60)
        
        # Should allow first requests
        for i in range(5):
            allowed, info = limiter.is_allowed(f"test_user_{int(time.time())}", limit=10, window=60)
            assert allowed is True
            assert info['remaining'] >= 0
    
    def test_rate_limiter_deny(self):
        """Test rate limiter denies requests over limit"""
        limiter = RateLimiter(default_limit=2, default_window=60)
        test_key = f"test_user_deny_{int(time.time())}"
        
        # Use up the limit
        allowed1, info1 = limiter.is_allowed(test_key, limit=2, window=60)
        assert allowed1 is True
        
        allowed2, info2 = limiter.is_allowed(test_key, limit=2, window=60)
        assert allowed2 is True
        
        # Should be denied now
        allowed3, info3 = limiter.is_allowed(test_key, limit=2, window=60)
        assert allowed3 is False
        assert info3['remaining'] == 0
    
    def test_sliding_window_allow(self):
        """Test sliding window allows requests within limit"""
        limiter = SlidingWindowRateLimiter(default_limit=10, default_window=60)
        
        # Should allow first requests
        for i in range(5):
            allowed, info = limiter.is_allowed(f"test_user_sw_{int(time.time())}", limit=10, window=60)
            assert allowed is True
    
    def test_sliding_window_deny(self):
        """Test sliding window denies requests over limit"""
        limiter = SlidingWindowRateLimiter(default_limit=2, default_window=60)
        test_key = f"test_user_sw_deny_{int(time.time())}"
        
        # Use up the limit
        allowed1, info1 = limiter.is_allowed(test_key, limit=2, window=60)
        assert allowed1 is True
        
        allowed2, info2 = limiter.is_allowed(test_key, limit=2, window=60)
        assert allowed2 is True
        
        # Should be denied now
        allowed3, info3 = limiter.is_allowed(test_key, limit=2, window=60)
        assert allowed3 is False
    
    def test_different_users_separate_limits(self):
        """Test that different users have separate rate limits"""
        limiter = RateLimiter(default_limit=1, default_window=60)
        timestamp = int(time.time())
        
        allowed1, _ = limiter.is_allowed(f"user1_{timestamp}", limit=1, window=60)
        assert allowed1 is True
        
        allowed2, _ = limiter.is_allowed(f"user2_{timestamp}", limit=1, window=60)
        assert allowed2 is True
        
        # Both users should now be limited
        denied1, _ = limiter.is_allowed(f"user1_{timestamp}", limit=1, window=60)
        assert denied1 is False
        
        denied2, _ = limiter.is_allowed(f"user2_{timestamp}", limit=1, window=60)
        assert denied2 is False


class TestFastAPIIntegration:
    """Test FastAPI middleware integration"""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter can be initialized"""
        limiter = get_rate_limiter()
        assert limiter is not None
    
    def test_cached_decorator_syntax(self):
        """Test that @fastapi_cached decorator can be applied"""
        @fastapi_cached(ttl=300)
        async def test_endpoint(request: Request):
            return {"status": "ok"}
        
        # Function should be callable
        assert callable(test_endpoint)
    
    def test_rate_limit_decorator_syntax(self):
        """Test that @fastapi_rate_limit decorator can be applied"""
        @fastapi_rate_limit(limit=10, window=60)
        async def test_endpoint(request: Request):
            return {"status": "ok"}
        
        # Function should be callable
        assert callable(test_endpoint)


class TestCachePerformance:
    """Test cache performance improvements"""
    
    @pytest.fixture
    def cache_client(self):
        """Create cache client"""
        return CacheClient()
    
    def test_cache_hit_performance(self, cache_client, benchmark):
        """Benchmark cache hit performance"""
        cache_client.set("perf_key", "perf_value")
        
        result = benchmark(cache_client.get, "perf_key")
        assert result == "perf_value"
    
    def test_cache_miss_performance(self, cache_client, benchmark):
        """Benchmark cache miss performance"""
        result = benchmark(cache_client.get, "nonexistent_key")
        assert result is None


class TestIntegration:
    """Integration tests for optimizations"""
    
    def test_cache_and_rate_limit_together(self):
        """Test cache and rate limiting work together"""
        cache = CacheClient()
        limiter = RateLimiter(default_limit=5, default_window=60)
        test_key = f"integration_user_{int(time.time())}"
        
        # Set cache value
        cache.set("test_integration", "value")
        
        # Check rate limiting
        for i in range(5):
            allowed, _ = limiter.is_allowed(test_key, limit=5, window=60)
            assert allowed is True
        
        # Get cached value
        value = cache.get("test_integration")
        assert value == "value"
        
        # Rate limit should now block
        denied, info = limiter.is_allowed(test_key, limit=5, window=60)
        assert denied is False
        assert info['remaining'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
