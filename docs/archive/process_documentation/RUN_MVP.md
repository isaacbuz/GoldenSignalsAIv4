# 🚀 GoldenSignalsAI MVP - Run Guide

## ✅ Phase 1 Implementation Complete!

All core components have been implemented:
- ✅ **4 Working Trading Agents** (RSI, MACD, Volume Spike, MA Crossover)
- ✅ **Meta Consensus Agent** (Combines signals from all agents)
- ✅ **Agent Orchestrator** (Manages all agents and generates signals)
- ✅ **Backend Integration** (FastAPI with real agent signals)
- ✅ **WebSocket Support** (Real-time signal streaming)
- ✅ **Frontend API Connection** (Endpoints match frontend expectations)

## 📦 What's Been Created

### Trading Agents
1. **`agents/core/technical/simple_working_agent.py`** - RSI Agent
2. **`agents/core/technical/macd_agent.py`** - MACD Agent
3. **`agents/core/technical/volume_spike_agent.py`** - Volume Spike Agent
4. **`agents/core/technical/ma_crossover_agent.py`** - Moving Average Crossover Agent

### Orchestration
5. **`agents/meta/simple_consensus_agent.py`** - Combines signals from multiple agents
6. **`agents/orchestration/simple_orchestrator.py`** - Manages all agents

### Backend Integration
7. **`src/main_simple_v2.py`** - Updated backend with orchestrator integration

## 🏃‍♂️ How to Run

### 1. Stop existing backend (if running)
```bash
# Find process on port 8000
lsof -i :8000
# Kill it
kill -9 [PID]
```

### 2. Start the new backend with agents
```bash
cd src
python main_simple_v2.py
```

### 3. Start frontend (if not running)
```bash
cd frontend
npm run dev
```

### 4. Access the system
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🧪 Test the System

### Test Backend Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get latest signals (what frontend uses)
curl http://localhost:8000/api/v1/signals/latest

# Get signal for specific symbol
curl http://localhost:8000/api/v1/signals/AAPL

# Get agent performance
curl http://localhost:8000/api/v1/agents/performance

# Get market data
curl http://localhost:8000/api/v1/market-data/AAPL
```

### Test WebSocket (using wscat)
```bash
# Install wscat if needed
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8000/ws/signals/AAPL
```

## 📊 What You Should See

### In the Terminal
```
🚀 Starting GoldenSignalsAI V3 with Agent Orchestrator...
✅ Market data service initialized
✅ Agent orchestrator initialized and started
Initialized 4 trading agents
Started signal generation with 300s interval
Generating initial signals...
Running signal generation cycle...
Generating signals for AAPL
Generating signals for GOOGL
...
```

### In the Frontend
- **Signal Stream**: Live signals from your agents
- **Agent Performance**: Real metrics from your agents
- **Market Pulse**: Real market data
- **Charts**: Live price data

## 🎯 Next Steps (Phase 2)

### Immediate Improvements
1. **Better Error Handling** - Graceful failures when market closed
2. **Signal Persistence** - Save signals to database
3. **Performance Tracking** - Track actual vs predicted performance
4. **More Agents** - Add Bollinger Bands, Stochastic, etc.

### Advanced Features
1. **Backtesting** - Test strategies on historical data
2. **Portfolio Management** - Track positions and P&L
3. **Risk Management** - Position sizing and stop losses
4. **Alert System** - Email/SMS notifications

### Infrastructure
1. **Database Integration** - PostgreSQL for signal history
2. **Redis Caching** - For real-time performance
3. **Docker Deployment** - Containerize everything
4. **Monitoring** - Grafana dashboards

## 🐛 Troubleshooting

### "Address already in use"
```bash
# Kill process on port 8000
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Frontend not showing signals?
1. Check browser console for errors
2. Verify backend is running: http://localhost:8000/health
3. Check API response: http://localhost:8000/api/v1/signals/latest

### Agents returning HOLD for everything?
- This can happen when market is closed or yfinance has issues
- Try during market hours (9:30 AM - 4:00 PM EST)
- Check agent logs in terminal for specific errors

### WebSocket not connecting?
- Make sure backend is running
- Check browser console for WebSocket errors
- Try refreshing the page

## 🎉 Congratulations!

You now have a working MVP with:
- Real trading agents generating signals
- Multi-agent consensus system
- Live market data integration
- WebSocket real-time updates
- Professional frontend dashboard

The system generates signals every 5 minutes using 4 different technical analysis strategies, combines them into consensus signals, and streams them to your frontend in real-time! 