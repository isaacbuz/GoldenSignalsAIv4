#!/usr/bin/env python3
'''Simple cache wrapper for immediate performance improvement'''

from functools import wraps
from cachetools import TTLCache
import time

# Create caches
market_cache = TTLCache(maxsize=1000, ttl=300)
signal_cache = TTLCache(maxsize=500, ttl=30)

def cache_market_data(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(symbol, *args, **kwargs):
            cache_key = f"{symbol}:{time.time() // ttl}"
            if cache_key in market_cache:
                return market_cache[cache_key]
            result = await func(symbol, *args, **kwargs)
            market_cache[cache_key] = result
            return result
        return wrapper
    return decorator

def cache_signals(ttl=30):
    def decorator(func):
        @wraps(func)
        async def wrapper(symbol, *args, **kwargs):
            cache_key = f"{symbol}:{time.time() // ttl}"
            if cache_key in signal_cache:
                return signal_cache[cache_key]
            result = await func(symbol, *args, **kwargs)
            signal_cache[cache_key] = result
            return result
        return wrapper
    return decorator
