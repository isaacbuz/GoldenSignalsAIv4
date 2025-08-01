#!/usr/bin/env python3
"""
Performance Optimization Script for GoldenSignalsAI
Applies immediate performance improvements to the existing backend
"""

import os
import sys
import json
from datetime import datetime

def create_optimized_backend():
    """Create an optimized version of the backend with caching"""

    optimizations = """
# Add these imports at the top of standalone_backend_fixed.py:
from functools import lru_cache
from cachetools import TTLCache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add cache configuration after imports:
market_data_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL
signal_cache = TTLCache(maxsize=500, ttl=30)  # 30 second TTL
executor = ThreadPoolExecutor(max_workers=4)

# Add caching to get_market_data function:
async def get_market_data_cached(symbol: str) -> Optional[MarketData]:
    cache_key = f"market:{symbol}"
    if cache_key in market_data_cache:
        return market_data_cache[cache_key]

    data = await get_market_data(symbol)
    if data:
        market_data_cache[cache_key] = data
    return data

# Add caching to generate_signals:
@lru_cache(maxsize=100)
def calculate_indicators_cached(symbol: str, prices_tuple: tuple) -> dict:
    # Convert tuple back to array for calculations
    prices = np.array(prices_tuple)
    return calculate_technical_indicators(prices)

# Optimize signal generation with batch processing:
async def generate_signals_batch(symbols: List[str]) -> List[Signal]:
    tasks = [generate_signals(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_signals = []
    for result in results:
        if isinstance(result, list):
            all_signals.extend(result)

    return all_signals

# Add response compression:
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add performance monitoring:
import time
from collections import defaultdict

request_times = defaultdict(list)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    request_times[request.url.path].append(process_time)
    return response

# Add performance stats endpoint:
@app.get("/api/v1/performance")
async def get_performance_stats():
    stats = {}
    for endpoint, times in request_times.items():
        if times:
            stats[endpoint] = {
                'count': len(times),
                'avg_ms': round(np.mean(times) * 1000, 2),
                'max_ms': round(max(times) * 1000, 2)
            }
    return stats
"""

    print("Performance Optimization Recommendations:")
    print("=" * 60)
    print(optimizations)
    print("=" * 60)

    # Create a simple cache wrapper
    cache_wrapper = """#!/usr/bin/env python3
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
"""

    with open("cache_wrapper.py", "w") as f:
        f.write(cache_wrapper)

    print("\nCreated cache_wrapper.py for immediate use")
    print("\nTo apply optimizations:")
    print("1. Add caching decorators to slow functions")
    print("2. Enable response compression")
    print("3. Implement batch processing for multiple requests")
    print("4. Use connection pooling for database/API calls")

def create_nginx_config():
    """Create optimized nginx configuration"""

    nginx_config = """# Optimized nginx configuration for GoldenSignalsAI
upstream backend {
    least_conn;
    server localhost:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name localhost;

    # Enable gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml application/json;

    # Cache static files
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API endpoints
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Enable caching for GET requests
        proxy_cache_methods GET HEAD;
        proxy_cache_valid 200 5m;
        proxy_cache_bypass $http_pragma $http_authorization;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
"""

    with open("nginx_optimized.conf", "w") as f:
        f.write(nginx_config)

    print("\nCreated nginx_optimized.conf for load balancing and caching")

def create_redis_config():
    """Create Redis configuration for caching"""

    redis_config = """# Redis configuration for GoldenSignalsAI caching

# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru

# Persistence (disable for pure caching)
save ""
appendonly no

# Performance tuning
tcp-backlog 511
timeout 0
tcp-keepalive 300

# Optimize for low latency
hz 10
dynamic-hz yes

# Enable pipelining
pipeline-flush-timeout 100
"""

    with open("redis_cache.conf", "w") as f:
        f.write(redis_config)

    print("\nCreated redis_cache.conf for high-performance caching")

def create_performance_test():
    """Create a performance testing script"""

    test_script = """#!/usr/bin/env python3
'''Performance testing script for GoldenSignalsAI'''

import asyncio
import aiohttp
import time
import statistics
from typing import List

async def test_endpoint(session: aiohttp.ClientSession, url: str, num_requests: int = 100):
    '''Test endpoint performance'''
    times = []

    for _ in range(num_requests):
        start = time.time()
        async with session.get(url) as response:
            await response.json()
        times.append(time.time() - start)

    return {
        'url': url,
        'requests': num_requests,
        'avg_ms': statistics.mean(times) * 1000,
        'median_ms': statistics.median(times) * 1000,
        'p95_ms': statistics.quantiles(times, n=20)[18] * 1000 if len(times) > 20 else max(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000
    }

async def run_performance_tests():
    '''Run performance tests on all endpoints'''
    base_url = "http://localhost:8000"
    endpoints = [
        "/api/v1/signals",
        "/api/v1/market-data/SPY",
        "/api/v1/signals/SPY/insights",
        "/api/v1/market/opportunities"
    ]

    async with aiohttp.ClientSession() as session:
        # Warm up
        for endpoint in endpoints:
            await session.get(f"{base_url}{endpoint}")

        # Run tests
        results = []
        for endpoint in endpoints:
            result = await test_endpoint(session, f"{base_url}{endpoint}")
            results.append(result)
            print(f"Tested {endpoint}: {result['avg_ms']:.2f}ms avg")

        # Test concurrent requests
        print("\\nTesting concurrent requests...")
        start = time.time()
        tasks = [session.get(f"{base_url}/api/v1/market-data/SPY") for _ in range(50)]
        await asyncio.gather(*tasks)
        concurrent_time = (time.time() - start) * 1000
        print(f"50 concurrent requests: {concurrent_time:.2f}ms total")

        return results

if __name__ == "__main__":
    print("Running performance tests...")
    results = asyncio.run(run_performance_tests())

    print("\\nPerformance Summary:")
    print("-" * 50)
    for result in results:
        print(f"{result['url']}:")
        print(f"  Average: {result['avg_ms']:.2f}ms")
        print(f"  P95: {result['p95_ms']:.2f}ms")
        print(f"  Range: {result['min_ms']:.2f}ms - {result['max_ms']:.2f}ms")
"""

    with open("test_performance.py", "w") as f:
        f.write(test_script)
    os.chmod("test_performance.py", 0o755)

    print("\nCreated test_performance.py for performance testing")

def main():
    """Main function"""
    print("GoldenSignalsAI Performance Optimization Tool")
    print("=" * 60)

    # Create optimization files
    create_optimized_backend()
    create_nginx_config()
    create_redis_config()
    create_performance_test()

    print("\n\nImmediate Optimization Steps:")
    print("1. Install Redis: brew install redis (macOS) or apt-get install redis-server (Linux)")
    print("2. Start Redis: redis-server redis_cache.conf")
    print("3. Apply caching decorators from cache_wrapper.py")
    print("4. Run performance tests: python test_performance.py")
    print("5. Monitor improvements with /api/v1/performance endpoint")

    print("\n\nExpected Performance Improvements:")
    print("- Market data requests: 166ms → <50ms (with caching)")
    print("- Signal generation: 1063ms → <200ms (with caching)")
    print("- Overall latency: 1671ms → <300ms")
    print("- Concurrent handling: 10x improvement")

    # Create a simple monitoring dashboard
    monitoring_html = """<!DOCTYPE html>
<html>
<head>
    <title>GoldenSignalsAI Performance Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { display: inline-block; margin: 20px; padding: 20px; border: 1px solid #ddd; }
        .chart-container { width: 600px; height: 400px; margin: 20px auto; }
    </style>
</head>
<body>
    <h1>Performance Monitor</h1>
    <div id="metrics"></div>
    <div class="chart-container">
        <canvas id="latencyChart"></canvas>
    </div>

    <script>
        async function updateMetrics() {
            const response = await fetch('http://localhost:8000/api/v1/performance');
            const data = await response.json();

            const metricsDiv = document.getElementById('metrics');
            metricsDiv.innerHTML = '';

            for (const [endpoint, stats] of Object.entries(data)) {
                if (typeof stats === 'object' && stats.avg_ms) {
                    const metric = document.createElement('div');
                    metric.className = 'metric';
                    metric.innerHTML = `
                        <h3>${endpoint}</h3>
                        <p>Requests: ${stats.count}</p>
                        <p>Avg: ${stats.avg_ms}ms</p>
                        <p>P95: ${stats.p95_ms || 'N/A'}ms</p>
                    `;
                    metricsDiv.appendChild(metric);
                }
            }
        }

        // Update every 5 seconds
        setInterval(updateMetrics, 5000);
        updateMetrics();
    </script>
</body>
</html>"""

    with open("performance_monitor.html", "w") as f:
        f.write(monitoring_html)

    print("\n\nCreated performance_monitor.html - Open in browser to monitor performance")

if __name__ == "__main__":
    main()
