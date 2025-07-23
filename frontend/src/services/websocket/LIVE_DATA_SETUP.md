# Live Data Setup Guide

## Backend Requirements

To connect live data to the chart, you need:

### 1. WebSocket Server
The backend (`src/main.py`) needs to be running with WebSocket endpoints:

```bash
# Start the backend server
cd ..  # Go to project root
python src/main.py
```

### 2. Data Sources
You can connect to various live data sources:

#### Option A: Alpha Vantage (Free tier available)
```python
# In your backend, add to .env:
ALPHA_VANTAGE_API_KEY=your_api_key

# The backend will fetch real-time quotes
```

#### Option B: Yahoo Finance (Free)
```python
# Already integrated in the backend
# Uses yfinance library for real-time data
```

#### Option C: Interactive Brokers API
```python
# For professional trading
# Requires IB Gateway or TWS running
```

### 3. Enable WebSocket in Frontend

In `ProfessionalChart.tsx`, uncomment the WebSocket integration:

```typescript
// Change this:
const isConnected = false; // Mock connection status

// To this:
const { isConnected, requestPrediction, requestSignals, requestPatterns } = useChartWebSocket({
  symbol,
  timeframe,
  onMarketData: (data: MarketData) => {
    // Real-time price updates
  },
  onPrediction: (data: PredictionData) => {
    // AI predictions
  },
  onSignal: (data: SignalData) => {
    // Trading signals
  },
  onPattern: (data: PatternData) => {
    // Pattern detection
  },
});
```

## Quick Start

### Step 1: Start the Backend
```bash
# From project root
pip install -r requirements.txt
python src/main.py
```

### Step 2: Configure Environment
Create `.env` in project root:
```
# API Keys (optional, for enhanced data)
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# WebSocket Configuration
WS_HOST=localhost
WS_PORT=8000
```

### Step 3: Update Frontend WebSocket URL
In `chartWebSocket.ts`:
```typescript
const WS_URL = process.env.NODE_ENV === 'production'
  ? 'wss://your-domain.com/ws'
  : 'ws://localhost:8000/ws';
```

### Step 4: Test Connection
The chart will automatically show:
- Green dot + "Live" when connected
- Yellow dot + "Demo Mode" when disconnected

## Data Flow

1. **Frontend** requests symbol data via WebSocket
2. **Backend** fetches from data provider (Yahoo Finance/Alpha Vantage)
3. **AI Agents** analyze the data in real-time
4. **Predictions** are generated and sent back
5. **Chart** updates with live prices, signals, and patterns

## Features When Connected

- Real-time price updates (every second)
- Live AI predictions updating as market moves
- Pattern detection alerts
- Multi-agent consensus signals
- Volume analysis
- Technical indicators calculated in real-time

## Troubleshooting

### Connection Failed
1. Check backend is running: `http://localhost:8000/health`
2. Check WebSocket endpoint: `ws://localhost:8000/ws`
3. Check browser console for CORS errors

### No Data
1. Verify symbol exists (e.g., "AAPL", not "APPL")
2. Check market hours (some APIs only work during trading hours)
3. Verify API keys are set correctly

### Performance Issues
1. Reduce update frequency in backend
2. Limit number of indicators
3. Use data batching for multiple symbols
