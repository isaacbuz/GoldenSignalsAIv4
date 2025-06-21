# 🚀 GoldenSignalsAI - Rate Limit Solutions Guide

## Overview

This guide provides comprehensive solutions for handling API rate limits in the GoldenSignalsAI system, particularly for Yahoo Finance and other market data providers.

## Rate Limit Challenges

### Yahoo Finance Limitations
- **Requests per minute**: ~60-100 (varies)
- **Requests per hour**: ~1,000-2,000
- **No official rate limit documentation**
- **429 errors when exceeded**
- **IP-based blocking for severe violations**

## Implemented Solutions

### 1. Multi-Level Caching Strategy

#### Memory Cache (TTL: 5-30 minutes)
```python
# Quotes: 5 minute cache
self.quote_cache = TTLCache(maxsize=1000, ttl=300)

# Historical data: 10 minute cache
self.historical_cache = TTLCache(maxsize=500, ttl=600)

# News: 30 minute cache
self.news_cache = TTLCache(maxsize=500, ttl=1800)
```

#### Redis Cache (TTL: 5-60 minutes)
- Distributed caching across instances
- Survives process restarts
- Shared between backend services

#### Disk Cache (TTL: 24 hours)
- Persistent storage for offline access
- Fallback during outages
- Historical data preservation

### 2. Request Throttling

#### Rate Limiter Implementation
```python
# Minimum interval between requests
min_interval_ms = 100  # 100ms = 10 requests/second max

# Per-minute limiting
requests_per_minute = 60

# Exponential backoff on errors
backoff_factor = 2.0
max_retries = 3
```

#### Smart Request Queue
- Priority-based processing (CRITICAL > HIGH > NORMAL > LOW)
- Batch processing for efficiency
- Automatic retry with backoff

### 3. Alternative Data Sources

#### Primary: Yahoo Finance
- Free, no API key required
- Good coverage of stocks
- Real-time and historical data

#### Fallback Options:
1. **Alpha Vantage**
   - Free tier: 5 requests/minute
   - Requires API key
   - Good for fundamental data

2. **IEX Cloud**
   - Free tier: 50,000 messages/month
   - Reliable and fast
   - Great documentation

3. **Polygon.io**
   - Free tier available
   - WebSocket support
   - Options data included

4. **Finnhub**
   - Free tier: 60 requests/minute
   - Real-time WebSocket
   - International markets

### 4. Batch Processing

#### Efficient Symbol Grouping
```python
# Batch size optimization
batch_size = 10  # Process 10 symbols at once

# Fetch all at once, cache individually
async def batch_get_quotes(symbols: List[str]):
    cached = check_cache(symbols)
    uncached = filter_uncached(symbols)
    
    if uncached:
        data = await fetch_batch(uncached)
        cache_results(data)
    
    return merge_results(cached, data)
```

## Configuration

### Environment Variables
```bash
# Redis for distributed caching
REDIS_URL=redis://localhost:6379

# Alternative data source API keys
ALPHA_VANTAGE_API_KEY=your_key_here
IEX_CLOUD_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
```

### Rate Limit Configuration
```python
rate_limits = {
    DataSource.YAHOO_FINANCE: RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        min_interval_ms=100
    ),
    DataSource.ALPHA_VANTAGE: RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=300,
        requests_per_day=500,
        min_interval_ms=12000
    )
}
```

## Usage Examples

### Basic Quote Fetching
```python
from src.services.rate_limit_handler import get_rate_limit_handler

# Get handler instance
handler = get_rate_limit_handler()

# Fetch single quote
quote = await handler.get_quote("AAPL")

# Fetch multiple quotes efficiently
quotes = await handler.batch_get_quotes(["AAPL", "GOOGL", "MSFT"])
```

### Historical Data
```python
# Fetch with automatic caching and rate limiting
hist_data = await handler.get_historical_data(
    symbol="AAPL",
    period="1d",
    interval="5m"
)
```

### Priority Requests
```python
# Critical request (e.g., active trading)
quote = await handler.get_quote(
    "AAPL", 
    priority=RequestPriority.CRITICAL
)

# Background request (e.g., research)
data = await handler.get_historical_data(
    "AAPL",
    priority=RequestPriority.LOW
)
```

## Best Practices

### 1. Cache First
- Always check cache before making API calls
- Use appropriate TTL for different data types
- Implement cache warming for popular symbols

### 2. Batch When Possible
- Group related requests together
- Use batch endpoints when available
- Process results in parallel

### 3. Handle Errors Gracefully
- Implement exponential backoff
- Provide fallback data sources
- Show cached/stale data with warnings

### 4. Monitor Usage
- Track request counts per source
- Log rate limit errors
- Alert on approaching limits

### 5. Optimize Refresh Rates
- Real-time data: 5-15 second intervals
- Minute data: 1-5 minute intervals
- Daily data: Cache for hours/days

## Monitoring and Debugging

### Check Rate Limit Status
```python
# View current request counts
handler = get_rate_limit_handler()
status = handler.get_rate_limit_status()
print(f"Yahoo Finance: {status['yahoo_finance']['requests_remaining']}/min")
```

### Debug Cache Hits
```python
# Enable debug logging
import logging
logging.getLogger('rate_limit_handler').setLevel(logging.DEBUG)
```

### Monitor Performance
```python
# Track response times
start = time.time()
quote = await handler.get_quote("AAPL")
elapsed = time.time() - start
print(f"Response time: {elapsed:.2f}s")
```

## Troubleshooting

### Common Issues

1. **429 Too Many Requests**
   - Solution: Increase min_interval_ms
   - Enable exponential backoff
   - Use alternative data sources

2. **Stale Data**
   - Solution: Reduce cache TTL
   - Implement cache invalidation
   - Add data freshness indicators

3. **Slow Response Times**
   - Solution: Increase cache size
   - Use Redis for faster access
   - Implement pre-fetching

4. **Missing Data**
   - Solution: Try multiple sources
   - Implement retry logic
   - Provide user feedback

## Future Enhancements

1. **WebSocket Integration**
   - Real-time data without polling
   - Reduced API calls
   - Lower latency

2. **Smart Caching**
   - ML-based cache TTL optimization
   - Predictive pre-fetching
   - Usage pattern analysis

3. **Distributed Rate Limiting**
   - Coordinate across multiple instances
   - Global rate limit tracking
   - Fair request distribution

4. **Premium Data Sources**
   - Direct exchange connections
   - Institutional data feeds
   - Dedicated infrastructure

## Conclusion

By implementing these rate limit solutions, GoldenSignalsAI can:
- Handle 10x more users without hitting limits
- Provide faster response times via caching
- Maintain service during API outages
- Scale efficiently as usage grows

The multi-layered approach ensures reliability while optimizing for performance and cost. 