# Performance Optimization Guide for GoldenSignalsAI V2

## Current Performance Issues

Based on the production data testing results:
- **Average Latency**: 1671ms (target: <500ms)
- **P95 Latency**: 1788ms
- **Signal Generation**: 1063ms
- **Market Data Fetch**: 166ms
- **Historical Data**: 64ms

## Performance Optimization Strategies

### 1. Caching Layer Implementation

#### Redis Caching
```python
import redis
import json
from datetime import timedelta
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_result(expiration=300):  # 5 minutes default
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

#### In-Memory Caching
```python
from cachetools import TTLCache
from functools import lru_cache

# Market data cache (5 minute TTL)
market_data_cache = TTLCache(maxsize=1000, ttl=300)

# Signal cache (30 second TTL for real-time updates)
signal_cache = TTLCache(maxsize=500, ttl=30)

@lru_cache(maxsize=128)
def calculate_indicators(symbol: str, data: tuple) -> dict:
    """Cache technical indicator calculations"""
    # Convert data back to array for calculations
    return compute_indicators(data)
```

### 2. Database Query Optimization

#### Connection Pooling
```python
from sqlalchemy.pool import QueuePool
from databases import Database

# Create connection pool
database = Database(
    DATABASE_URL,
    min_size=10,
    max_size=20,
    command_timeout=10,
    pool_recycle=3600
)

# Use async queries
async def get_historical_data(symbol: str, start_date: datetime):
    query = """
        SELECT * FROM market_data 
        WHERE symbol = :symbol AND timestamp >= :start_date
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    return await database.fetch_all(query, values={
        "symbol": symbol,
        "start_date": start_date
    })
```

#### Query Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_signals_symbol_timestamp ON signals(symbol, timestamp DESC);
CREATE INDEX idx_signals_action ON signals(action);

-- Materialized view for signal insights
CREATE MATERIALIZED VIEW signal_insights AS
SELECT 
    symbol,
    action,
    COUNT(*) as signal_count,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as last_signal
FROM signals
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY symbol, action;

-- Refresh every 5 minutes
CREATE OR REPLACE FUNCTION refresh_signal_insights()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY signal_insights;
END;
$$ LANGUAGE plpgsql;
```

### 3. Async Processing Optimization

#### Concurrent Data Fetching
```python
import asyncio
from typing import List, Dict

async def fetch_market_data_batch(symbols: List[str]) -> Dict[str, MarketData]:
    """Fetch market data for multiple symbols concurrently"""
    tasks = [fetch_single_market_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    market_data = {}
    for symbol, result in zip(symbols, results):
        if not isinstance(result, Exception):
            market_data[symbol] = result
    
    return market_data

async def fetch_single_market_data(symbol: str) -> MarketData:
    # Check cache first
    cached = market_data_cache.get(symbol)
    if cached:
        return cached
    
    # Fetch from API
    data = await get_market_data_from_api(symbol)
    market_data_cache[symbol] = data
    return data
```

#### Background Task Processing
```python
from fastapi import BackgroundTasks
import asyncio

class SignalProcessor:
    def __init__(self):
        self.processing_queue = asyncio.Queue()
        self.results_cache = {}
    
    async def process_signals_background(self):
        """Process signals in the background"""
        while True:
            try:
                task = await self.processing_queue.get()
                result = await self._process_signal(task)
                self.results_cache[task['id']] = result
            except Exception as e:
                logger.error(f"Background processing error: {e}")
            await asyncio.sleep(0.1)
    
    async def submit_signal_task(self, symbol: str, data: dict):
        task_id = f"{symbol}_{int(time.time())}"
        await self.processing_queue.put({
            'id': task_id,
            'symbol': symbol,
            'data': data
        })
        return task_id
```

### 4. ML Model Optimization

#### Model Loading and Caching
```python
import joblib
from functools import lru_cache

class OptimizedMLSignalGenerator:
    def __init__(self):
        self._models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models once at startup"""
        model_files = ['momentum', 'mean_reversion', 'trend_following']
        for model_name in model_files:
            self._models[model_name] = joblib.load(f'models/{model_name}.pkl')
    
    @lru_cache(maxsize=256)
    def predict_cached(self, features_hash: str, model_name: str):
        """Cache predictions for identical feature sets"""
        features = self._decode_features(features_hash)
        return self._models[model_name].predict(features)
```

#### Batch Prediction
```python
async def generate_signals_batch(symbols: List[str]) -> List[Signal]:
    """Generate signals for multiple symbols in batch"""
    # Fetch all data concurrently
    market_data = await fetch_market_data_batch(symbols)
    
    # Prepare features for all symbols
    all_features = []
    symbol_indices = []
    
    for i, symbol in enumerate(symbols):
        if symbol in market_data:
            features = extract_features(market_data[symbol])
            all_features.append(features)
            symbol_indices.append(i)
    
    # Batch prediction
    if all_features:
        predictions = ml_model.predict_batch(np.array(all_features))
        
        # Create signals
        signals = []
        for idx, pred in zip(symbol_indices, predictions):
            signal = create_signal(symbols[idx], pred, market_data[symbols[idx]])
            signals.append(signal)
        
        return signals
    
    return []
```

### 5. API Response Optimization

#### Response Compression
```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

#### Pagination
```python
from fastapi import Query
from typing import Optional

@app.get("/api/v1/signals")
async def get_signals(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = None
):
    # Use database pagination
    query = "SELECT * FROM signals"
    if symbol:
        query += f" WHERE symbol = '{symbol}'"
    query += f" ORDER BY timestamp DESC LIMIT {limit} OFFSET {skip}"
    
    return await database.fetch_all(query)
```

#### Field Selection
```python
@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    fields: Optional[List[str]] = Query(None)
):
    full_data = await fetch_market_data(symbol)
    
    if fields:
        # Return only requested fields
        return {field: full_data.get(field) for field in fields}
    
    return full_data
```

### 6. WebSocket Optimization

#### Throttling and Batching
```python
class OptimizedWebSocketManager:
    def __init__(self):
        self.connections = set()
        self.update_queue = asyncio.Queue()
        self.batch_interval = 0.5  # 500ms batching
    
    async def batch_sender(self):
        """Send updates in batches"""
        while True:
            updates = []
            deadline = time.time() + self.batch_interval
            
            # Collect updates for batch_interval seconds
            while time.time() < deadline:
                try:
                    update = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=deadline - time.time()
                    )
                    updates.append(update)
                except asyncio.TimeoutError:
                    break
            
            if updates:
                # Send batch to all connections
                await self.broadcast_batch(updates)
```

### 7. Infrastructure Optimization

#### Load Balancing
```nginx
upstream backend {
    least_conn;
    server backend1:8000 weight=5;
    server backend2:8000 weight=5;
    keepalive 32;
}

server {
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

#### CDN for Static Assets
```yaml
# CloudFront configuration
frontend:
  cdn:
    enabled: true
    providers:
      - cloudfront:
          distribution_id: "E1234567890"
          cache_behaviors:
            - path_pattern: "*.js"
              ttl: 86400
            - path_pattern: "*.css"
              ttl: 86400
```

### 8. Monitoring and Profiling

#### Performance Monitoring
```python
import time
from prometheus_client import Histogram, Counter

# Metrics
request_duration = Histogram('request_duration_seconds', 'Request duration')
request_count = Counter('request_count', 'Request count', ['method', 'endpoint'])

@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_duration.observe(duration)
    request_count.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    return response
```

#### Profiling
```python
import cProfile
import pstats

def profile_endpoint(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return result
    return wrapper
```

## Implementation Priority

1. **Immediate (Week 1)**
   - Implement Redis caching for market data and signals
   - Add database connection pooling
   - Enable response compression

2. **Short-term (Week 2-3)**
   - Optimize ML model loading and prediction
   - Implement batch processing for signals
   - Add WebSocket throttling

3. **Medium-term (Month 1-2)**
   - Set up load balancing
   - Implement background task processing
   - Add comprehensive monitoring

4. **Long-term (Month 3+)**
   - Migrate to microservices architecture
   - Implement edge caching
   - Add auto-scaling

## Expected Performance Improvements

With these optimizations:
- **Average Latency**: 1671ms → <300ms (82% improvement)
- **Signal Generation**: 1063ms → <200ms (81% improvement)
- **Concurrent Users**: 100 → 1000+ (10x improvement)
- **Throughput**: 60 req/min → 600+ req/min (10x improvement)

## Monitoring Success

Track these KPIs:
- P50, P95, P99 latencies
- Requests per second
- Error rates
- Cache hit rates
- CPU and memory usage
- Database query times

Use tools like:
- Prometheus + Grafana
- New Relic or DataDog
- Custom dashboards
- Load testing with Locust or K6
