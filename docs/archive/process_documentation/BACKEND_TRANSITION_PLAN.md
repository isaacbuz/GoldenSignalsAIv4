# Backend Transition Plan: Simple → Production

## Current State
- **Simple Backend** (`simple_backend.py`): Currently running with mock data
- **Frontend**: Running on port 3000, working with simple backend
- **Databases**: PostgreSQL and Redis configured but not actively used
- **Live Data**: Connection module created but not integrated

## Production Backend Overview
The main backend (`src/main.py`) includes:
- Real-time data from multiple sources (yfinance, Alpha Vantage, Polygon)
- PostgreSQL for persistent storage
- Redis for caching and real-time data
- WebSocket support for live updates
- Multi-agent trading system
- Authentication and rate limiting
- Monitoring (Sentry, Prometheus)

## Transition Steps

### Phase 1: Prepare Environment (Current)
✅ **Completed:**
- Simple backend running and tested
- Frontend working with mock data
- Database connections configured
- Live data connector created

🔄 **In Progress:**
- Testing all endpoints
- Verifying frontend compatibility

### Phase 2: Database Setup
```bash
# 1. Create .env file from example
cp env.example .env

# 2. Edit .env with your credentials
# - Database passwords
# - API keys (Alpha Vantage, Polygon, etc.)

# 3. Start databases
docker-compose up -d postgres redis

# 4. Run database migrations
alembic upgrade head

# 5. Verify connections
python check_databases.py
```

### Phase 3: Install Missing Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Additional production dependencies
pip install sentry-sdk prometheus-client alembic

# WebSocket support
pip install websocket-client python-socketio
```

### Phase 4: Test Production Backend
```bash
# 1. Stop simple backend
pkill -f "simple_backend.py"

# 2. Start production backend in test mode
python main.py --test

# 3. Run integration tests
python tests/integration/test_production_backend.py
```

### Phase 5: Data Migration
```python
# Run the training data preparation
python prepare_full_training_data.py

# This will:
# - Fetch 20 years of historical data
# - Store in PostgreSQL
# - Calculate technical indicators
# - Prepare ML training datasets
```

### Phase 6: Gradual Transition
```bash
# 1. Run both backends on different ports
python simple_backend.py --port 8001 &  # Backup
python main.py --port 8000 &            # Production

# 2. Update frontend to use production endpoints
# 3. Monitor for issues
# 4. Once stable, stop simple backend
```

## Configuration Changes

### Frontend Updates
```typescript
// Update API base URL in frontend config
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Add WebSocket support
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
```

### Environment Variables
```env
# Required for production
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/goldensignals
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
ALPHA_VANTAGE_API_KEY=your-key
POLYGON_API_KEY=your-key
SENTRY_DSN=your-sentry-dsn
```

## Feature Comparison

| Feature | Simple Backend | Production Backend |
|---------|---------------|-------------------|
| Market Data | Mock | Real-time (multiple sources) |
| Signals | Random | ML-based multi-agent |
| Storage | Memory | PostgreSQL + Redis |
| WebSocket | Basic | Full duplex with rooms |
| Auth | None | JWT tokens |
| Monitoring | Logs | Sentry + Prometheus |
| Rate Limiting | None | Per-user limits |
| Backtesting | None | 20 years historical |

## Rollback Plan
If issues arise during transition:

1. **Immediate Rollback**:
   ```bash
   # Stop production backend
   pkill -f "main.py"
   
   # Restart simple backend
   python simple_backend.py &
   ```

2. **Data Preservation**:
   - All data in PostgreSQL is preserved
   - Redis cache can be cleared without data loss
   - Frontend continues working with either backend

## Success Criteria
- [ ] All endpoints return correct data
- [ ] Response times < 100ms for most endpoints
- [ ] WebSocket connections stable
- [ ] No frontend errors
- [ ] Live data updating correctly
- [ ] ML models generating quality signals
- [ ] Database queries optimized
- [ ] Error rate < 0.1%

## Timeline
- **Day 1**: Environment setup and testing
- **Day 2**: Database migration and data loading
- **Day 3**: Production backend testing
- **Day 4**: Frontend integration
- **Day 5**: Monitoring and optimization
- **Day 6-7**: Observation and fine-tuning

## Next Steps
1. Review and approve this plan
2. Set up production environment variables
3. Begin Phase 2 (Database Setup)
4. Schedule transition window
5. Notify team of changes

## Commands Quick Reference
```bash
# Start simple backend
python simple_backend.py

# Start production backend
python main.py

# Run tests
python test_backend_endpoints.py

# Check system status
./start.sh status

# View logs
./start.sh logs backend

# Full system start (production)
./start.sh prod
``` 