# 🗄️ GoldenSignalsAI V3 - Database Setup Guide

## Overview

GoldenSignalsAI requires two main databases for proper operation:

1. **PostgreSQL** - Primary relational database for structured data
2. **Redis** - In-memory cache and real-time data streaming

## Database Requirements

### 1. PostgreSQL (Required)
- **Purpose**: Store signals, user data, agent performance, historical data
- **Version**: 14+ (15 recommended)
- **Extensions**: TimescaleDB (optional, for time-series optimization)
- **Storage**: ~50GB initially, grows with usage

### 2. Redis (Required)
- **Purpose**: Real-time data caching, WebSocket state, pub/sub messaging
- **Version**: 7.0+
- **Memory**: 2-4GB minimum
- **Persistence**: AOF recommended for production

## Quick Setup Options

### Option 1: Docker Compose (Recommended for Development)

The easiest way to set up all required databases:

```bash
# From the project root directory
docker-compose up -d database redis

# This will start:
# - PostgreSQL on port 5432
# - Redis on port 6379
```

### Option 2: Local Installation

#### PostgreSQL Setup

**macOS:**
```bash
# Install PostgreSQL
brew install postgresql@15
brew services start postgresql@15

# Create database and user
createdb goldensignals
psql goldensignals -c "CREATE USER goldensignals WITH PASSWORD 'your_secure_password';"
psql goldensignals -c "GRANT ALL PRIVILEGES ON DATABASE goldensignals TO goldensignals;"
```

**Ubuntu/Debian:**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createdb goldensignals
sudo -u postgres psql -c "CREATE USER goldensignals WITH PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE goldensignals TO goldensignals;"
```

#### Redis Setup

**macOS:**
```bash
# Install Redis
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
# Install Redis
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Option 3: Cloud Services (Production)

#### AWS
- **PostgreSQL**: Amazon RDS (db.r6g.xlarge recommended)
- **Redis**: Amazon ElastiCache

#### Azure
- **PostgreSQL**: Azure Database for PostgreSQL
- **Redis**: Azure Cache for Redis

#### Google Cloud
- **PostgreSQL**: Cloud SQL
- **Redis**: Memorystore

## Database Schema

### PostgreSQL Tables

The system will automatically create these tables on first run:

```sql
-- Trading signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(36) UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    strength VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    source VARCHAR(50) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agent performance tracking
CREATE TABLE agent_performance (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(36) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    total_signals INTEGER DEFAULT 0,
    correct_signals INTEGER DEFAULT 0,
    accuracy FLOAT DEFAULT 0.0,
    avg_confidence FLOAT DEFAULT 0.0,
    current_weight FLOAT DEFAULT 1.0,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- User accounts (if authentication is enabled)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    roles TEXT[] DEFAULT ARRAY['user'],
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Market data cache
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price FLOAT NOT NULL,
    high_price FLOAT NOT NULL,
    low_price FLOAT NOT NULL,
    close_price FLOAT NOT NULL,
    volume BIGINT NOT NULL,
    indicators JSONB,
    UNIQUE(symbol, timestamp)
);

-- Create indexes for performance
CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_signals_created_at ON signals(created_at);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# PostgreSQL Configuration
DATABASE_URL=postgresql+asyncpg://goldensignals:your_secure_password@localhost:5432/goldensignals
DB_HOST=localhost
DB_PORT=5432
DB_NAME=goldensignals
DB_USER=goldensignals
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Optional, leave empty for local development

# Optional: Use SQLite for development (no PostgreSQL needed)
# DATABASE_URL=sqlite+aiosqlite:///./goldensignals.db
```

### Testing Database Connection

Run this script to test your database setup:

```python
# test_db_connection.py
import asyncio
import asyncpg
import redis
from sqlalchemy.ext.asyncio import create_async_engine

async def test_connections():
    # Test PostgreSQL
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='goldensignals',
            password='your_secure_password',
            database='goldensignals'
        )
        await conn.fetchval('SELECT 1')
        await conn.close()
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
    
    # Test Redis
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connections())
```

## Database Initialization

The system will automatically initialize the database schema on first run. You can also manually initialize:

```bash
# Run database migrations
python -m alembic upgrade head

# Or use the setup script
python scripts/setup_database.py
```

## Backup and Maintenance

### PostgreSQL Backup

```bash
# Backup
pg_dump -U goldensignals -h localhost goldensignals > backup_$(date +%Y%m%d).sql

# Restore
psql -U goldensignals -h localhost goldensignals < backup_20240115.sql
```

### Redis Backup

```bash
# Save snapshot
redis-cli BGSAVE

# Copy dump file
cp /var/lib/redis/dump.rdb backup_$(date +%Y%m%d).rdb
```

## Performance Optimization

### PostgreSQL Tuning

Add to `postgresql.conf`:
```ini
# Memory
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

# Connections
max_connections = 200

# Write performance
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### Redis Tuning

Add to `redis.conf`:
```ini
# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
appendonly yes
appendfsync everysec

# Performance
tcp-keepalive 60
timeout 300
```

## Monitoring

### Health Checks

The system provides health check endpoints:

```bash
# Check all services
curl http://localhost:8000/health

# Response includes database status:
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  ...
}
```

### Database Metrics

Monitor these key metrics:
- **PostgreSQL**: Connection count, query performance, disk usage
- **Redis**: Memory usage, hit rate, eviction count

## Troubleshooting

### Common Issues

1. **"Connection refused" error**
   - Check if services are running: `ps aux | grep postgres` or `ps aux | grep redis`
   - Verify ports are not blocked by firewall

2. **"Authentication failed"**
   - Double-check credentials in `.env` file
   - Ensure user has proper permissions

3. **"Database does not exist"**
   - Create database: `createdb goldensignals`

4. **Redis "OOM" errors**
   - Increase memory limit or enable eviction policy
   - Check memory usage: `redis-cli INFO memory`

## Development vs Production

### Development (SQLite Option)

For quick development without PostgreSQL:

```bash
# In .env file
DATABASE_URL=sqlite+aiosqlite:///./goldensignals.db

# This creates a local SQLite file
# Note: Some features may be limited
```

### Production Requirements

- **PostgreSQL**: Dedicated instance with replication
- **Redis**: Cluster mode with persistence
- **Backups**: Automated daily backups
- **Monitoring**: Prometheus + Grafana
- **High Availability**: Multi-zone deployment

## Next Steps

1. ✅ Set up databases using one of the options above
2. ✅ Configure environment variables
3. ✅ Test connections
4. ✅ Start the application
5. ✅ Monitor health endpoint

The system will handle all table creation and schema management automatically on startup! 