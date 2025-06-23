# GoldenSignalsAI V2 Architecture Consolidation Plan

## Executive Summary

The current codebase has significant duplication:
- **4 agent directories** with 281 total files
- **4 backend implementations** doing similar work
- **7+ entry points** causing confusion
- **Scattered configuration** across multiple locations

This plan consolidates everything into a clean, unified architecture while preserving the best functionality.

## Current State Analysis 🔍

### 1. Agent Implementations (CRITICAL DUPLICATION)

| Directory | Files | Size | Purpose | Status |
|-----------|-------|------|---------|---------|
| `agents/` | 173 | 1.8M | Most comprehensive, recent updates | **KEEP & CONSOLIDATE** |
| `src/agents/` | 49 | 880K | Older orchestration logic | **MERGE VALUABLE** |
| `archive/legacy_backend_agents/` | 58 | 248K | Legacy implementations | **ARCHIVE** |
| `src/domain/trading/agents/` | 1 | - | Empty structure | **DELETE** |

### 2. Backend Files (CRITICAL DUPLICATION)

| File | Lines | Purpose | Decision |
|------|-------|---------|----------|
| `standalone_backend_optimized.py` | 1,056 | Latest with caching | **KEEP AS REFERENCE** |
| `standalone_backend_fixed.py` | ~800 | Bug fixes | **EXTRACT FIXES** |
| `standalone_backend.py` | ~800 | Original | **ARCHIVE** |
| `simple_backend.py` | ~900 | Alternative impl | **ARCHIVE** |

### 3. Entry Points (CONFUSING)

| File | Purpose | Decision |
|------|---------|----------|
| `src/main.py` | Should be primary | **MAKE PRIMARY** |
| `main.py` | Root duplicate | **DELETE** |
| `src/main_v2.py` | Alternative version | **MERGE & DELETE** |
| `src/main_simple.py` | Simple version | **ARCHIVE** |
| `start_backend.py` | Starter script | **CONVERT TO SHELL** |
| `start_simple.py` | Simple starter | **DELETE** |
| `start_daily_work.py` | Utility | **KEEP IN SCRIPTS/** |

### 4. Configuration (SCATTERED)

| Location | Purpose | Decision |
|----------|---------|----------|
| `src/core/config.py` | Main config | **PRIMARY CONFIG** |
| `config/` | YAML configs | **MERGE TO src/config/** |
| `src/config/` | App config | **MERGE WITH CORE** |
| `.env` files | Environment | **CONSOLIDATE** |

## Consolidated Architecture 🏗️

```
GoldenSignalsAI_V2/
├── src/                          # All source code
│   ├── main.py                   # SINGLE ENTRY POINT
│   ├── api/                      # API Layer
│   │   └── v1/
│   │       ├── signals.py
│   │       ├── market_data.py
│   │       ├── monitoring.py
│   │       ├── backtest.py
│   │       └── agents.py         # NEW: Agent management
│   ├── services/                 # Business Logic
│   │   ├── market_data_manager.py # NEW: Multi-provider
│   │   ├── signal_service.py
│   │   ├── agent_orchestrator.py # CONSOLIDATED
│   │   └── ...
│   ├── agents/                   # UNIFIED AGENTS
│   │   ├── base/
│   │   │   ├── trading_agent.py # NEW: Standard base
│   │   │   └── registry.py      # Agent registry
│   │   ├── technical/           # From agents/core/technical/
│   │   ├── sentiment/           # From agents/core/sentiment/
│   │   ├── options/             # From agents/core/options/
│   │   ├── portfolio/           # From agents/core/portfolio/
│   │   └── ml/                  # Best ML agents
│   ├── core/
│   │   ├── config.py           # SINGLE CONFIG
│   │   ├── database.py
│   │   └── dependencies.py
│   └── utils/
├── scripts/                     # All scripts
│   ├── start.sh                # Main startup
│   ├── test.sh
│   └── deploy.sh
├── tests/                       # All tests
├── archive/                     # Old implementations
│   ├── legacy_backends/
│   ├── legacy_agents/
│   └── legacy_configs/
└── docs/                        # Documentation
```

## Week 1: Critical Fixes & Data Provider 🚨

### Day 1-2: Fix Data Provider (BLOCKER)
```bash
# 1. Install dependencies
pip install requests-cache

# 2. Integrate market_data_manager.py
# Already created - just needs integration

# 3. Update standalone_backend_optimized.py to use it
# Follow MARKET_DATA_INTEGRATION_GUIDE.md
```

### Day 3-4: Consolidate Entry Points
```python
# src/main.py - Single entry point
from fastapi import FastAPI
from src.api.v1 import signals, market_data, monitoring, backtest, agents
from src.config.config import settings
from src.services.market_data_manager import get_market_data_manager

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="GoldenSignalsAI V2",
        version="2.0.0",
        docs_url="/api/docs"
    )
    
    # Initialize services
    app.state.market_data = get_market_data_manager()
    
    # Include routers
    app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
    app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["market"])
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
    app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtest"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
    
    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Day 5: Emergency Archival
```bash
# Archive duplicates
mkdir -p archive/legacy_backends
mv standalone_backend.py simple_backend.py archive/legacy_backends/
mv main.py src/main_v2.py src/main_simple.py archive/

# Create startup script
cat > scripts/start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
python src/main.py
EOF
chmod +x scripts/start.sh
```

## Week 2: Unify Architecture & Add Monitoring 🏗️

### Day 6-7: Agent Consolidation

#### Step 1: Create Unified Base Agent
```python
# src/agents/base/trading_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
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
        self._performance_metrics = {
            'signals_generated': 0,
            'successful_signals': 0,
            'average_confidence': 0.0,
            'last_signal_time': None
        }
    
    @abstractmethod
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze market data and generate signal"""
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        pass
    
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before sending"""
        if signal.confidence < self.config.get('min_confidence', 0.6):
            return False
        if not 0 <= signal.confidence <= 1:
            return False
        if signal.action not in ['BUY', 'SELL', 'HOLD']:
            return False
        return True
    
    def update_metrics(self, signal: Signal, success: bool = None):
        """Update performance metrics"""
        self._performance_metrics['signals_generated'] += 1
        if success is not None and success:
            self._performance_metrics['successful_signals'] += 1
        self._performance_metrics['last_signal_time'] = datetime.now()
```

#### Step 2: Migrate Best Agents
```python
# Priority agents to migrate (based on recent updates and functionality):
PRIORITY_AGENTS = [
    # Options
    'gamma_exposure_agent.py',      # Most recent, sophisticated
    'precise_options_signals.py',   # High-value options analysis
    
    # Technical
    'hybrid_pattern_agent.py',       # Advanced pattern recognition
    'hybrid_bollinger_agent.py',     # Bollinger bands with ML
    'hybrid_macd_agent.py',          # MACD with enhancements
    'hybrid_rsi_agent.py',           # RSI with ML
    'hybrid_volume_agent.py',        # Volume analysis
    
    # Sentiment
    'hybrid_sentiment_flow_agent.py', # Sentiment + flow
    
    # ML/AI
    'enhanced_ml_meta_agent.py',     # Meta learning
    
    # Portfolio
    'portfolio_management_ai.py',     # AI-driven portfolio
    
    # Arbitrage
    'arbitrage_signals.py',          # Cross-market opportunities
]
```

#### Step 3: Create Agent Registry
```python
# src/agents/base/registry.py
from typing import Dict, Type, List
from agents.base.trading_agent import TradingAgent
import importlib
import logging

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Central registry for all trading agents"""
    
    def __init__(self):
        self._agents: Dict[str, Type[TradingAgent]] = {}
        self._instances: Dict[str, TradingAgent] = {}
        
    def register(self, name: str, agent_class: Type[TradingAgent]):
        """Register an agent class"""
        if name in self._agents:
            logger.warning(f"Overwriting agent {name}")
        self._agents[name] = agent_class
        logger.info(f"Registered agent: {name}")
        
    def create_agent(self, name: str, config: Dict = None) -> TradingAgent:
        """Create an agent instance"""
        if name not in self._agents:
            raise ValueError(f"Unknown agent: {name}")
            
        if name not in self._instances:
            self._instances[name] = self._agents[name](name, config)
            
        return self._instances[name]
    
    def get_all_agents(self) -> List[str]:
        """Get list of all registered agents"""
        return list(self._agents.keys())
    
    def auto_discover(self, path: str = "src.agents"):
        """Auto-discover and register agents"""
        # Implementation to scan and load agents
        pass

# Global registry
agent_registry = AgentRegistry()
```

### Day 8-9: Service Layer Consolidation

#### Unified Orchestrator
```python
# src/services/agent_orchestrator.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from agents.base.registry import agent_registry
from agents.base.trading_agent import Signal
from src.services.market_data_manager import get_market_data_manager
import logging

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates multiple agents for signal generation"""
    
    def __init__(self):
        self.market_data = get_market_data_manager()
        self.active_agents: List[str] = []
        self.consensus_threshold = 0.6
        
    async def initialize_agents(self, agent_names: List[str] = None):
        """Initialize selected agents or all available"""
        if agent_names is None:
            agent_names = agent_registry.get_all_agents()
            
        self.active_agents = agent_names
        logger.info(f"Initialized {len(self.active_agents)} agents")
        
    async def generate_signals(self, symbol: str) -> List[Signal]:
        """Generate signals from all active agents"""
        # Get market data
        market_data = await self.market_data.get_market_data(symbol)
        historical = await self.market_data.get_historical_data(symbol)
        
        # Prepare data package
        data_package = {
            'symbol': symbol,
            'current': market_data,
            'historical': historical,
            'timestamp': datetime.now()
        }
        
        # Run all agents in parallel
        tasks = []
        for agent_name in self.active_agents:
            agent = agent_registry.create_agent(agent_name)
            tasks.append(self._run_agent(agent, symbol, data_package))
            
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid signals
        signals = []
        for result in results:
            if isinstance(result, Signal):
                signals.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Agent error: {result}")
                
        return signals
    
    async def _run_agent(self, agent: TradingAgent, symbol: str, data: Dict) -> Optional[Signal]:
        """Run a single agent with error handling"""
        try:
            signal = await agent.analyze(symbol, data)
            if signal and await agent.validate_signal(signal):
                return signal
        except Exception as e:
            logger.error(f"Error in {agent.name}: {e}")
            return None
    
    async def get_consensus_signals(self, symbol: str) -> List[Signal]:
        """Get signals with consensus filtering"""
        all_signals = await self.generate_signals(symbol)
        
        # Group by action
        action_groups = {}
        for signal in all_signals:
            if signal.action not in action_groups:
                action_groups[signal.action] = []
            action_groups[signal.action].append(signal)
        
        # Filter by consensus
        consensus_signals = []
        for action, signals in action_groups.items():
            if len(signals) / len(self.active_agents) >= self.consensus_threshold:
                # Create consensus signal
                avg_confidence = sum(s.confidence for s in signals) / len(signals)
                consensus_signal = Signal(
                    id=f"consensus_{symbol}_{action}_{datetime.now().timestamp()}",
                    agent_name="consensus",
                    symbol=symbol,
                    action=action,
                    confidence=avg_confidence,
                    price=signals[0].price,  # Use first signal's price
                    timestamp=datetime.now(),
                    reasoning={'agents': [s.agent_name for s in signals]},
                    indicators={},
                    risk_metrics={}
                )
                consensus_signals.append(consensus_signal)
                
        return consensus_signals
```

### Day 10: Monitoring & Observability

```python
# src/services/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time
import logging

# Metrics
api_requests = Counter(
    'goldensignals_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

signal_generation_duration = Histogram(
    'goldensignals_signal_generation_seconds',
    'Time to generate signals',
    ['agent', 'symbol']
)

active_agents = Gauge(
    'goldensignals_active_agents',
    'Number of active agents'
)

market_data_errors = Counter(
    'goldensignals_market_data_errors_total',
    'Market data fetch errors',
    ['provider', 'error_type']
)

def track_performance(metric_name: str):
    """Decorator to track function performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                signal_generation_duration.labels(
                    agent=kwargs.get('agent_name', 'unknown'),
                    symbol=kwargs.get('symbol', 'unknown')
                ).observe(duration)
                return result
            except Exception as e:
                market_data_errors.labels(
                    provider='unknown',
                    error_type=type(e).__name__
                ).inc()
                raise
        return wrapper
    return decorator

# Health check endpoint
async def health_check():
    """System health check"""
    checks = {
        'database': await check_database(),
        'redis': await check_redis(),
        'market_data': await check_market_data(),
        'agents': len(agent_registry.get_all_agents()) > 0
    }
    
    status = 'healthy' if all(checks.values()) else 'unhealthy'
    return {
        'status': status,
        'checks': checks,
        'timestamp': datetime.now().isoformat()
    }
```

## Week 3: Testing & Production Readiness 🧪

### Day 11-12: Integration Testing

```python
# tests/integration/test_unified_system.py
import pytest
from httpx import AsyncClient
from src.main import create_app
from agents.base.registry import agent_registry

@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_agents():
    """Register mock agents for testing"""
    # Register test agents
    pass

class TestUnifiedSystem:
    async def test_market_data_fallback(self, client):
        """Test market data manager fallback mechanism"""
        response = await client.get("/api/v1/market-data/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert 'provider' in data  # Shows which provider was used
        
    async def test_agent_orchestration(self, client, mock_agents):
        """Test multiple agents working together"""
        response = await client.get("/api/v1/signals?symbol=AAPL")
        assert response.status_code == 200
        signals = response.json()
        assert len(signals) > 0
        
    async def test_consensus_signals(self, client, mock_agents):
        """Test consensus signal generation"""
        response = await client.get("/api/v1/signals/consensus?symbol=AAPL")
        assert response.status_code == 200
```

### Day 13-14: Performance Testing

```python
# tests/performance/test_load.py
import asyncio
import aiohttp
import time

async def test_concurrent_requests():
    """Test system under load"""
    url = "http://localhost:8000/api/v1/signals"
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] * 20  # 100 requests
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [fetch_signal(session, url, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
    successful = sum(1 for r in results if r)
    print(f"Completed {successful}/100 requests in {duration:.2f}s")
    print(f"Requests per second: {successful/duration:.2f}")
```

### Day 15: Final Cleanup & Documentation

```bash
# Final directory structure
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Archive old implementations
mkdir -p archive/pre_consolidation
mv agents archive/pre_consolidation/old_agents
mv src/agents archive/pre_consolidation/old_src_agents

# Move consolidated agents
mv consolidated_agents src/agents

# Update documentation
echo "# GoldenSignalsAI V2 - Consolidated Architecture

## Quick Start
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
./scripts/start.sh
\`\`\`

## Architecture
- Single entry point: src/main.py
- Unified agents in src/agents/
- Multi-provider market data
- Prometheus monitoring
" > README_CONSOLIDATED.md
```

## Migration Checklist ✅

### Week 1: Critical Fixes
- [ ] Install requests-cache
- [ ] Integrate market_data_manager.py
- [ ] Update backend to use new data manager
- [ ] Create single src/main.py entry point
- [ ] Archive duplicate backends
- [ ] Create scripts/start.sh

### Week 2: Architecture
- [ ] Create unified TradingAgent base class
- [ ] Implement AgentRegistry
- [ ] Migrate top 12 priority agents
- [ ] Create AgentOrchestrator service
- [ ] Add Prometheus monitoring
- [ ] Implement health checks

### Week 3: Testing
- [ ] Integration tests for unified system
- [ ] Performance/load testing
- [ ] Final cleanup and archival
- [ ] Update all documentation
- [ ] Deploy to staging environment

## Benefits of Consolidation 🎯

1. **Reduced Complexity**: From 281 agent files to ~50 best implementations
2. **Clear Architecture**: Single entry point, clear service layers
3. **Better Performance**: Unified caching, parallel agent execution
4. **Easier Testing**: Standardized agent interface
5. **Production Ready**: Monitoring, health checks, error handling
6. **Maintainable**: Clear separation of concerns

## Risk Mitigation 🛡️

1. **Full Archival**: Nothing is deleted, just moved to archive/
2. **Gradual Migration**: Can run old and new systems in parallel
3. **Comprehensive Testing**: Each component tested before migration
4. **Rollback Plan**: Git tags at each major milestone
5. **Performance Monitoring**: Metrics to catch any degradation

## Success Metrics 📊

- **Code Reduction**: 70% fewer files to maintain
- **Performance**: < 100ms signal generation (P95)
- **Reliability**: 99.9% uptime with fallbacks
- **Test Coverage**: > 80% for critical paths
- **Developer Experience**: New developer onboarding < 1 day

This consolidation will transform GoldenSignalsAI from a complex prototype into a production-ready trading platform. 