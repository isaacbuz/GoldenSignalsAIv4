# GoldenSignalsAI V2 Troubleshooting Guide

## Table of Contents

1. [Common Issues](#common-issues)
2. [API & Data Issues](#api--data-issues)
3. [Backend Issues](#backend-issues)
4. [Frontend Issues](#frontend-issues)
5. [Database Issues](#database-issues)
6. [Performance Issues](#performance-issues)
7. [Deployment Issues](#deployment-issues)
8. [Debugging Tools](#debugging-tools)
9. [Log Analysis](#log-analysis)
10. [Getting Help](#getting-help)

## Common Issues

### Issue: Application Won't Start

**Symptoms:**
- Server crashes on startup
- Import errors
- Configuration errors

**Solutions:**

1. **Check Python Version**
```bash
python --version  # Should be 3.9+
```

2. **Verify Virtual Environment**
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Verify activation
which python  # Should point to .venv
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For development
```

4. **Check Environment Variables**
```bash
# Verify .env file exists
ls -la .env

# Check required variables
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('DATABASE_URL'))"
```

### Issue: Import Errors

**Symptoms:**
```python
ModuleNotFoundError: No module named 'src'
ImportError: cannot import name 'SignalGenerationEngine'
```

**Solutions:**

1. **Fix Python Path**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or in .env file
PYTHONPATH=/path/to/GoldenSignalsAI_V2
```

2. **Install in Development Mode**
```bash
pip install -e .
```

3. **Check Module Structure**
```bash
# Verify __init__.py files exist
find . -name "__init__.py" | grep src
```

## API & Data Issues

### Issue: yfinance HTTP 401 Errors

**Symptoms:**
```
ERROR:__main__:Error fetching market data for AAPL: HTTP Error 401
```

**Solutions:**

1. **Use Direct yfinance API**
```python
# In standalone_backend_optimized.py
ticker = yf.Ticker(symbol)
info = ticker.info  # Don't use fast_info
```

2. **Enable Fallback Sources**
```bash
# Set API keys in .env
ALPHA_VANTAGE_API_KEY=your-key
IEX_CLOUD_API_KEY=your-key
POLYGON_API_KEY=your-key
```

3. **Clear yfinance Cache**
```bash
# Clear cache directory
rm -rf ~/.cache/py-yfinance/
```

4. **Use Mock Data (Development)**
```python
# Enable mock data in development
MOCK_DATA_ENABLED=True
```

### Issue: No Market Data Returned

**Symptoms:**
- Empty responses from market data endpoints
- "No data found" errors

**Solutions:**

1. **Check Market Hours**
```python
# Verify market is open
from src.utils.timezone_utils import is_market_hours
print(f"Market open: {is_market_hours()}")
```

2. **Verify Symbol Validity**
```bash
# Test symbol directly
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info.get('regularMarketPrice'))"
```

3. **Check Data Source Priority**
```python
# In rate_limit_handler.py, verify source order
sources = handler._get_source_priority()
print(f"Data sources: {[s.value for s in sources]}")
```

## Backend Issues

### Issue: High Memory Usage

**Symptoms:**
- Server consuming excessive RAM
- Memory errors
- Slow performance

**Solutions:**

1. **Limit Worker Processes**
```python
# In gunicorn_config.py
workers = 2  # Reduce from 4
worker_connections = 500  # Reduce from 1000
```

2. **Enable Memory Profiling**
```bash
# Install memory profiler
pip install memory-profiler

# Run with profiling
mprof run python standalone_backend_optimized.py
mprof plot
```

3. **Clear Caches Periodically**
```python
# Add cache cleanup task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cache_cleanup_task())
```

### Issue: Slow API Response Times

**Symptoms:**
- Endpoints taking > 1 second
- Timeouts
- Poor user experience

**Solutions:**

1. **Enable Redis Caching**
```bash
# Install Redis
sudo apt install redis-server

# Start Redis
redis-server

# Configure in .env
REDIS_URL=redis://localhost:6379/0
```

2. **Add Database Indexes**
```sql
-- Improve query performance
CREATE INDEX idx_signals_symbol_timestamp ON signals(symbol, timestamp DESC);
CREATE INDEX idx_outcomes_signal_id ON signal_outcomes(signal_id);
```

3. **Use Batch Endpoints**
```python
# Instead of multiple calls
for symbol in symbols:
    data = await get_quote(symbol)

# Use batch endpoint
data = await batch_get_quotes(symbols)
```

## Frontend Issues

### Issue: Frontend Build Fails

**Symptoms:**
- npm errors during build
- Module resolution failures
- TypeScript errors

**Solutions:**

1. **Clean Install Dependencies**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

2. **Check Node Version**
```bash
node --version  # Should be 16.x+
nvm use 16      # If using nvm
```

3. **Fix TypeScript Errors**
```bash
# Run type check
npm run type-check

# Fix lint errors
npm run lint:fix
```

### Issue: WebSocket Connection Fails

**Symptoms:**
- "WebSocket connection failed"
- Real-time updates not working

**Solutions:**

1. **Check WebSocket URL**
```javascript
// In frontend code
const ws = new WebSocket('ws://localhost:8000/ws');
// For production: wss://your-domain.com/ws
```

2. **Verify Nginx Configuration**
```nginx
location /ws {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

3. **Check CORS Settings**
```python
# In backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Database Issues

### Issue: Database Connection Errors

**Symptoms:**
- "Could not connect to database"
- SQLite locked errors
- PostgreSQL connection refused

**Solutions:**

1. **SQLite Locked**
```python
# Enable WAL mode for better concurrency
conn.execute("PRAGMA journal_mode=WAL")
```

2. **PostgreSQL Connection**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify connection
psql -U goldensignals -d goldensignals -h localhost

# Check pg_hba.conf
sudo nano /etc/postgresql/14/main/pg_hba.conf
# Ensure: local all all md5
```

3. **Connection Pool Exhaustion**
```python
# Increase pool size
DATABASE_URL = "postgresql://user:pass@localhost/db?pool_size=20&max_overflow=0"
```

### Issue: Migration Failures

**Symptoms:**
- Database schema out of sync
- Missing tables/columns

**Solutions:**

1. **Manual Schema Update**
```sql
-- Check current schema
SELECT * FROM sqlite_master WHERE type='table';

-- Add missing columns
ALTER TABLE signals ADD COLUMN quality_score REAL DEFAULT 0.0;
```

2. **Recreate Tables**
```python
# Backup data first!
from src.services.signal_monitoring_service import SignalMonitoringService
service = SignalMonitoringService()
service._init_database()  # Recreates tables
```

## Performance Issues

### Issue: High CPU Usage

**Symptoms:**
- 100% CPU utilization
- Server unresponsive
- Slow calculations

**Solutions:**

1. **Profile CPU Usage**
```bash
# Install profiler
pip install py-spy

# Profile running process
py-spy top --pid $(pgrep -f standalone_backend)
```

2. **Optimize Indicator Calculations**
```python
# Cache technical indicators
@lru_cache(maxsize=1000)
def calculate_indicators(symbol, period):
    # Expensive calculations
    pass
```

3. **Use Async Operations**
```python
# Convert sync to async
async def calculate_all_indicators(symbols):
    tasks = [calculate_indicators(s) for s in symbols]
    return await asyncio.gather(*tasks)
```

### Issue: Memory Leaks

**Symptoms:**
- Memory usage grows over time
- Eventually crashes with OOM

**Solutions:**

1. **Identify Leaks**
```python
import tracemalloc
tracemalloc.start()

# ... run application ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6:.1f} MB")
```

2. **Fix Common Leaks**
```python
# Clear caches periodically
def clear_old_cache_entries():
    cutoff = datetime.now() - timedelta(hours=1)
    for key in list(cache.keys()):
        if cache[key]['timestamp'] < cutoff:
            del cache[key]
```

## Deployment Issues

### Issue: Docker Build Fails

**Symptoms:**
- Docker image build errors
- Dependency conflicts

**Solutions:**

1. **Multi-stage Build**
```dockerfile
# Use multi-stage to reduce size
FROM python:3.9-slim as builder
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
```

2. **Fix Permission Issues**
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser
```

### Issue: Kubernetes Pod CrashLoopBackOff

**Symptoms:**
- Pods constantly restarting
- Application crashes in container

**Solutions:**

1. **Check Pod Logs**
```bash
kubectl logs -f pod-name -n goldensignals
kubectl describe pod pod-name -n goldensignals
```

2. **Increase Resources**
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

3. **Add Health Checks**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Debugging Tools

### 1. API Testing

```bash
# Test endpoints with curl
curl -X GET http://localhost:8000/api/v1/signals | jq

# Test with httpie
http GET localhost:8000/api/v1/signals

# Load test with locust
locust -f locustfile.py --host=http://localhost:8000
```

### 2. Database Inspection

```bash
# SQLite
sqlite3 data/goldensignals.db
.tables
.schema signals
SELECT COUNT(*) FROM signals;

# PostgreSQL
psql -U goldensignals -d goldensignals
\dt  -- List tables
\d+ signals  -- Describe table
```

### 3. Process Monitoring

```bash
# System resources
htop
iotop

# Python specific
py-spy top --pid $(pgrep -f standalone_backend)

# Network connections
netstat -tulpn | grep 8000
lsof -i :8000
```

## Log Analysis

### 1. Application Logs

```bash
# Tail logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log | tail -20

# Count errors by type
grep -i error logs/app.log | cut -d' ' -f5- | sort | uniq -c | sort -nr
```

### 2. Log Aggregation

```python
# Add structured logging
import structlog

logger = structlog.get_logger()
logger.info("signal_generated", 
    symbol="AAPL", 
    confidence=0.85,
    action="BUY"
)
```

### 3. Monitoring Dashboards

```yaml
# Grafana dashboard query for error rate
rate(http_requests_total{status=~"5.."}[5m])

# Alert on high error rate
alert: HighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
for: 5m
```

## Getting Help

### 1. Diagnostic Information

When reporting issues, include:

```bash
# System info
uname -a
python --version
pip freeze > requirements-actual.txt

# Error logs
tail -n 100 logs/error.log

# Configuration (sanitized)
grep -v PASSWORD .env
```

### 2. Debug Mode

```bash
# Enable debug logging
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with verbose output
python standalone_backend_optimized.py --debug
```

### 3. Support Channels

1. **GitHub Issues**: 
   - Use issue templates
   - Include reproduction steps
   - Attach relevant logs

2. **Community Discord**: 
   - Real-time help
   - Share experiences
   - Feature discussions

3. **Email Support**: 
   - support@goldensignals.ai
   - Include diagnostic information
   - Expected response: 24-48 hours

### 4. Emergency Contacts

For production emergencies:
- **On-call Engineer**: +1-XXX-XXX-XXXX
- **Escalation**: engineering-lead@goldensignals.ai
- **Status Page**: https://status.goldensignals.ai

## Prevention Best Practices

1. **Regular Maintenance**
   - Update dependencies monthly
   - Review logs weekly
   - Monitor metrics continuously

2. **Testing**
   - Run tests before deployment
   - Load test new features
   - Monitor after releases

3. **Documentation**
   - Keep troubleshooting guide updated
   - Document new issues and solutions
   - Share knowledge with team 