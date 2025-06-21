#!/usr/bin/env python3
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
        print("\nTesting concurrent requests...")
        start = time.time()
        tasks = [session.get(f"{base_url}/api/v1/market-data/SPY") for _ in range(50)]
        await asyncio.gather(*tasks)
        concurrent_time = (time.time() - start) * 1000
        print(f"50 concurrent requests: {concurrent_time:.2f}ms total")
        
        return results

if __name__ == "__main__":
    print("Running performance tests...")
    results = asyncio.run(run_performance_tests())
    
    print("\nPerformance Summary:")
    print("-" * 50)
    for result in results:
        print(f"{result['url']}:")
        print(f"  Average: {result['avg_ms']:.2f}ms")
        print(f"  P95: {result['p95_ms']:.2f}ms")
        print(f"  Range: {result['min_ms']:.2f}ms - {result['max_ms']:.2f}ms")
