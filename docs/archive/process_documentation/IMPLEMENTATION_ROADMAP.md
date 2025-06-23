# GoldenSignalsAI V2 Implementation Roadmap

## Overview
Transform the current complex prototype into a production-ready trading platform by consolidating 1000+ files into ~300 well-organized components.

## Week 1: Critical Fixes & Data Provider

### Day 1: Emergency Cleanup & Data Provider Fix ⚡

#### Morning: Backup & Archive
```bash
# Create backup
cp -r . ../GoldenSignalsAI_V2_backup_$(date +%Y%m%d)

# Archive duplicate backends
mkdir -p archive/2024-06-legacy/backends
mv standalone_backend.py standalone_backend_fixed.py simple_backend.py archive/2024-06-legacy/backends/

# Archive duplicate entry points
mkdir -p archive/2024-06-legacy/scripts
mv start_backend.py start_simple.py archive/2024-06-legacy/scripts/
```

#### Afternoon: Integrate Market Data Manager
```bash
# Install dependencies
pip install requests-cache

# Test the new market data manager
python -c "
from src.services.market_data_manager import get_market_data_manager
import asyncio

async def test():
    mdm = get_market_data_manager()
    data = await mdm.get_market_data('AAPL')
    print(f'Got data from {data['provider']}: ${data['price']}')

asyncio.run(test())
"
```

### Day 2: Consolidate Entry Point

#### Create Unified Main Entry
```python
# src/main.py
import os
os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.v1 import signals, market_data, monitoring, agents
from src.config.config import settings
from src.services.market_data_manager import get_market_data_manager
from src.services.agent_orchestrator import AgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting GoldenSignalsAI V2...")
    app.state.market_data = get_market_data_manager()
    app.state.orchestrator = AgentOrchestrator()
    await app.state.orchestrator.initialize_agents()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="GoldenSignalsAI V2",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["market"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])

@app.get("/")
async def root():
    return {"message": "GoldenSignalsAI V2 API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Day 3: Documentation Cleanup

```bash
# Organize documentation
mkdir -p docs/{guides,implementation,planning,archive}

# Move files (preview first)
ls *GUIDE.md
ls *IMPLEMENTATION*.md
ls *PLAN*.md *ROADMAP*.md
ls *SUMMARY*.md *COMPLETE*.md

# Actually move them
mv *GUIDE.md docs/guides/ 2>/dev/null
mv *IMPLEMENTATION*.md docs/implementation/ 2>/dev/null
mv *PLAN*.md *ROADMAP*.md docs/planning/ 2>/dev/null
mv *SUMMARY*.md *COMPLETE*.md docs/archive/ 2>/dev/null

# Keep only essential in root
# README.md, LICENSE.md, CONTRIBUTING.md, .env.example
```

### Day 4-5: Update Backend Integration

#### Update standalone_backend_optimized.py to use new manager:
```python
# At the top
from src.services.market_data_manager import get_market_data_manager

# Replace initialization
market_data_manager = get_market_data_manager()

# Replace get_market_data_cached
async def get_market_data_cached(symbol: str) -> Optional[MarketData]:
    try:
        data = await market_data_manager.get_market_data(symbol)
        return MarketData(
            symbol=data['symbol'],
            price=data['price'],
            change=0,
            change_percent=0,
            volume=data.get('volume', 0),
            timestamp=data['timestamp'].isoformat(),
            high=data.get('high', data['price']),
            low=data.get('low', data['price']),
            open=data['price']
        )
    except Exception as e:
        logger.error(f"Market data error: {e}")
        # Return mock data to keep system running
        return generate_mock_market_data(symbol)
```

## Week 2: Unify Architecture & Add Monitoring

### Day 6: Create Base Agent Framework

```bash
# Create new agent structure
mkdir -p src/agents/{base,technical,sentiment,options,portfolio,ml}

# Create base agent
cat > src/agents/base/trading_agent.py << 'EOF'
"""Base Trading Agent Framework"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from cachetools import TTLCache

@dataclass
class Signal:
    """Unified signal format"""
    id: str
    agent_name: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: datetime
    reasoning: Dict[str, Any]
    indicators: Dict[str, float]
    risk_metrics: Dict[str, float]

class TradingAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.cache = TTLCache(maxsize=100, ttl=60)
    
    @abstractmethod
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze market data and generate signal"""
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        pass
EOF
```

### Day 7: Migrate Priority Agents

```bash
# List of priority agents to migrate
PRIORITY_AGENTS=(
    "gamma_exposure_agent.py"
    "precise_options_signals.py"
    "hybrid_pattern_agent.py"
    "hybrid_bollinger_agent.py"
    "hybrid_macd_agent.py"
    "hybrid_rsi_agent.py"
    "portfolio_management_ai.py"
)

# Copy and adapt each
for agent in "${PRIORITY_AGENTS[@]}"; do
    echo "Migrating $agent..."
    # Find and copy (implement actual migration logic)
done
```

### Day 8: Create Agent Registry

```python
# src/agents/base/registry.py
from typing import Dict, Type, List
import logging

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Central registry for all trading agents"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
            cls._instance._instances = {}
        return cls._instance
    
    def register(self, name: str, agent_class: Type[TradingAgent]):
        """Register an agent class"""
        self._agents[name] = agent_class
        logger.info(f"Registered agent: {name}")
    
    def create_agent(self, name: str, config: Dict = None) -> TradingAgent:
        """Create or get agent instance"""
        if name not in self._agents:
            raise ValueError(f"Unknown agent: {name}")
        
        if name not in self._instances:
            self._instances[name] = self._agents[name](name, config)
        
        return self._instances[name]
    
    def get_all_agents(self) -> List[str]:
        """Get list of all registered agents"""
        return list(self._agents.keys())

# Global registry instance
agent_registry = AgentRegistry()

# Auto-register agents
def register_agent(name: str):
    """Decorator to auto-register agents"""
    def decorator(cls):
        agent_registry.register(name, cls)
        return cls
    return decorator
```

### Day 9: Implement Agent Orchestrator

```python
# src/services/agent_orchestrator.py
from typing import List, Dict, Any
import asyncio
from agents.base.registry import agent_registry
from agents.base.trading_agent import Signal

class AgentOrchestrator:
    """Orchestrates multiple agents for signal generation"""
    
    def __init__(self):
        self.active_agents = []
        self.consensus_threshold = 0.6
    
    async def initialize_agents(self, agent_names: List[str] = None):
        """Initialize selected agents"""
        if agent_names is None:
            agent_names = agent_registry.get_all_agents()
        
        self.active_agents = agent_names
        logger.info(f"Initialized {len(self.active_agents)} agents")
    
    async def generate_signals(self, symbol: str) -> List[Signal]:
        """Generate signals from all active agents"""
        # Implementation from consolidation plan
        pass
```

### Day 10: Add Monitoring

```bash
# Install Prometheus client
pip install prometheus-client

# Create monitoring service
cat > src/services/monitoring.py << 'EOF'
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
api_requests = Counter(
    'goldensignals_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

signal_generation_time = Histogram(
    'goldensignals_signal_generation_seconds',
    'Time to generate signals',
    ['agent', 'symbol']
)

active_agents = Gauge(
    'goldensignals_active_agents',
    'Number of active agents'
)
EOF
```

## Week 3: Testing & Production Readiness

### Day 11-12: Comprehensive Testing

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Integration tests
pytest tests/integration/ -v

# Performance tests
python tests/performance/test_load.py
```

### Day 13-14: Final Cleanup

```bash
# Remove all __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Archive old implementations
mkdir -p archive/2024-06-legacy
mv agents archive/2024-06-legacy/old_agents
mv src/agents archive/2024-06-legacy/old_src_agents

# Update .gitignore
cat >> .gitignore << 'EOF'
# Build artifacts
*.pyc
__pycache__/
.coverage
htmlcov/
*.db
*.dylib

# IDE
.vscode/
.idea/

# Environment
.env
venv/
.venv/
EOF
```

### Day 15: Production Deployment

```bash
# Create production script
cat > scripts/start_production.sh << 'EOF'
#!/bin/bash
set -e

# Load environment
source .env.production

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check health
sleep 10
curl -f http://localhost:8000/health || exit 1

echo "GoldenSignalsAI V2 is running!"
EOF

chmod +x scripts/start_production.sh
```

## Success Criteria

### Week 1 ✓
- [ ] No more yfinance 401 errors
- [ ] Single entry point working
- [ ] Documentation organized
- [ ] Backend using new data manager

### Week 2 ✓
- [ ] Unified agent architecture
- [ ] Agent registry working
- [ ] Top agents migrated
- [ ] Monitoring enabled

### Week 3 ✓
- [ ] All tests passing
- [ ] Performance validated
- [ ] Clean project structure
- [ ] Production ready

## Quick Validation Tests

```bash
# Test 1: Data provider working
curl http://localhost:8000/api/v1/market-data/AAPL

# Test 2: Signals generating
curl http://localhost:8000/api/v1/signals?symbol=AAPL

# Test 3: Agent status
curl http://localhost:8000/api/v1/agents

# Test 4: Health check
curl http://localhost:8000/health

# Test 5: Metrics
curl http://localhost:8000/metrics
```

## Emergency Rollback

If anything goes wrong:
```bash
# Restore from backup
cp -r ../GoldenSignalsAI_V2_backup_$(date +%Y%m%d)/* .

# Or use git
git checkout main
git reset --hard HEAD
```

## Final Result

After following this roadmap:
- **70% code reduction** (1000+ files → 300 files)
- **Single source of truth** for each component
- **Production-ready** with monitoring and health checks
- **Clear architecture** that new developers can understand
- **Reliable data** with fallback providers

The system will be ready for rigorous testing and production deployment. 