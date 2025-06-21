#!/usr/bin/env python3
"""
Test script for rate limit handler
Demonstrates caching, throttling, and fallback mechanisms
"""

import asyncio
import time
from datetime import datetime
from src.services.rate_limit_handler import get_rate_limit_handler, RequestPriority

async def test_single_quote():
    """Test fetching a single quote with caching"""
    print("\n🔍 Testing single quote fetch...")
    handler = get_rate_limit_handler()
    
    # First fetch - will hit API
    start = time.time()
    quote1 = await handler.get_quote("AAPL")
    time1 = time.time() - start
    print(f"✅ First fetch: {time1:.2f}s - Price: ${quote1.get('price', 0) if quote1 else 'N/A'}")
    
    # Second fetch - should be cached
    start = time.time()
    quote2 = await handler.get_quote("AAPL")
    time2 = time.time() - start
    print(f"✅ Cached fetch: {time2:.2f}s - Price: ${quote2.get('price', 0) if quote2 else 'N/A'}")
    
    print(f"⚡ Cache speedup: {time1/time2:.1f}x faster")

async def test_batch_quotes():
    """Test batch quote fetching"""
    print("\n📊 Testing batch quote fetch...")
    handler = get_rate_limit_handler()
    
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    start = time.time()
    quotes = await handler.batch_get_quotes(symbols)
    elapsed = time.time() - start
    
    print(f"✅ Fetched {len(quotes)} quotes in {elapsed:.2f}s")
    for symbol, quote in quotes.items():
        if quote:
            print(f"  {symbol}: ${quote.get('price', 0)}")

async def test_rate_limiting():
    """Test rate limiting behavior"""
    print("\n⏱️ Testing rate limiting...")
    handler = get_rate_limit_handler()
    
    # Try to make rapid requests
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM"]
    
    print("Making rapid requests...")
    for i, symbol in enumerate(symbols):
        start = time.time()
        quote = await handler.get_quote(symbol, priority=RequestPriority.HIGH)
        elapsed = time.time() - start
        
        if quote:
            print(f"  Request {i+1}: {symbol} - {elapsed:.2f}s - ${quote.get('price', 0)}")
        else:
            print(f"  Request {i+1}: {symbol} - {elapsed:.2f}s - FAILED")
        
        # Small delay to show rate limiting
        await asyncio.sleep(0.05)

async def test_historical_data():
    """Test historical data fetching with caching"""
    print("\n📈 Testing historical data fetch...")
    handler = get_rate_limit_handler()
    
    # First fetch
    start = time.time()
    hist1 = await handler.get_historical_data("AAPL", period="5d", interval="1h")
    time1 = time.time() - start
    
    if hist1 is not None:
        print(f"✅ First fetch: {time1:.2f}s - {len(hist1)} records")
    else:
        print(f"❌ First fetch failed")
    
    # Cached fetch
    start = time.time()
    hist2 = await handler.get_historical_data("AAPL", period="5d", interval="1h")
    time2 = time.time() - start
    
    if hist2 is not None:
        print(f"✅ Cached fetch: {time2:.2f}s - {len(hist2)} records")
        print(f"⚡ Cache speedup: {time1/time2:.1f}x faster")

async def test_priority_queue():
    """Test request prioritization"""
    print("\n🎯 Testing request prioritization...")
    handler = get_rate_limit_handler()
    
    # Mix of priorities
    requests = [
        ("AAPL", RequestPriority.CRITICAL),
        ("GOOGL", RequestPriority.LOW),
        ("MSFT", RequestPriority.HIGH),
        ("AMZN", RequestPriority.NORMAL),
        ("TSLA", RequestPriority.CRITICAL),
    ]
    
    tasks = []
    for symbol, priority in requests:
        task = handler.get_quote(symbol, priority=priority)
        tasks.append((symbol, priority, task))
    
    print("Submitted requests with different priorities...")
    
    # Wait for all to complete
    for symbol, priority, task in tasks:
        quote = await task
        if quote:
            print(f"  {symbol} ({priority.name}): ${quote.get('price', 0)}")

async def show_cache_stats():
    """Display cache statistics"""
    print("\n📊 Cache Statistics:")
    handler = get_rate_limit_handler()
    
    # Memory cache stats
    for cache_type, cache in handler.memory_cache.items():
        print(f"  {cache_type}: {len(cache)} items cached")
    
    # Disk cache stats
    import os
    if os.path.exists(handler.disk_cache_dir):
        files = os.listdir(handler.disk_cache_dir)
        print(f"  Disk cache: {len(files)} files")

async def main():
    """Run all tests"""
    print("🚀 GoldenSignalsAI Rate Limit Handler Test")
    print("=" * 50)
    
    try:
        # Run tests
        await test_single_quote()
        await test_batch_quotes()
        await test_rate_limiting()
        await test_historical_data()
        await test_priority_queue()
        await show_cache_stats()
        
        print("\n✅ All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 