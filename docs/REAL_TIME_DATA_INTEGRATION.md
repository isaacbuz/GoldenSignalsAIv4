# Real-Time Data Integration Guide

## Overview

We have successfully enhanced the real-time data integration between the frontend charts and the Python backend, ensuring live market data flows seamlessly to the UI.

## Architecture

### Data Flow
1. **Market Data Service** (`backendMarketDataService.ts`)
   - Fetches current market prices
   - Retrieves historical OHLC data
   - Gets AI trading signals
   - Manages data caching

2. **Chart Component** (`ProfessionalChart.tsx`)
   - Uses market data service for all data
   - Falls back to mock data if backend unavailable
   - Updates in real-time via WebSocket

3. **WebSocket Integration** (`ChartSignalAgent`)
   - Handles real-time price updates
   - Streams AI signal updates
   - Manages connection lifecycle

## Backend Endpoints

### Market Data
```
GET /api/v1/market-data/{symbol}
Response: {
  symbol: string,
  price: number,
  change: number,
  change_percent: number,
  volume?: number,
  timestamp: string
}
```

### Historical Data
```
GET /api/v1/market-data/{symbol}/history?period=30d&interval=1d
Response: {
  symbol: string,
  period: string,
  interval: string,
  data: [{
    time: number,
    open: number,
    high: number,
    low: number,
    close: number,
    volume: number
  }]
}
```

### AI Signals
```
GET /api/v1/signals/symbol/{symbol}
Response: {
  signals: [{
    id: string,
    symbol: string,
    action: 'BUY' | 'SELL' | 'HOLD',
    confidence: number,
    price: number,
    timestamp: string,
    agents_consensus: {
      agentsInFavor: number,
      totalAgents: number
    }
  }]
}
```

### WebSocket
```
WS /ws/market-data
Messages:
- Subscribe: { type: 'subscribe', symbol: 'AAPL' }
- Price Update: { type: 'price', symbol: 'AAPL', price: 150.25, volume: 1000000 }
- Signal Update: { type: 'signal', symbol: 'AAPL', action: 'BUY', confidence: 0.85 }
```

## Frontend Implementation

### Service Configuration
```typescript
// backendMarketDataService.ts
const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### Usage in Charts
```typescript
// Fetch current price
const marketData = await fetchMarketData(symbol);

// Fetch historical data
const historicalData = await fetchHistoricalData(symbol, timeframe);

// Fetch AI signals
const signals = await fetchSignals(symbol);

// WebSocket connection (handled by ChartSignalAgent)
connectWebSocket(symbol, (data) => {
  // Handle real-time updates
});
```

## Features

### 1. Data Caching
- 30-second cache for market data
- Reduces API calls
- Improves performance

### 2. Fallback Mechanism
- Returns mock data if backend unavailable
- Ensures charts always display
- Smooth user experience

### 3. Real-time Updates
- WebSocket for live prices
- Automatic reconnection
- Heartbeat mechanism

### 4. Timeframe Mapping
```typescript
'1m' → period: '1d', interval: '1m'
'5m' → period: '5d', interval: '5m'
'1h' → period: '1mo', interval: '1h'
'1d' → period: '1y', interval: '1d'
```

## Environment Configuration

Create a `.env` file in the frontend directory:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENABLE_MOCK_DATA=false
VITE_ENABLE_WEBSOCKET=true
```

## Testing the Integration

1. **Start the Backend**
   ```bash
   cd /path/to/project
   python src/main.py
   ```

2. **Start the Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Verify Data Flow**
   - Open browser developer tools
   - Check Network tab for API calls
   - Monitor Console for data logs
   - Verify WebSocket connection in WS tab

## Troubleshooting

### No Data Displayed
1. Check backend is running on port 8000
2. Verify CORS settings in backend
3. Check browser console for errors
4. Ensure `.env` file exists with correct API URL

### WebSocket Not Connecting
1. Check WebSocket endpoint in backend
2. Verify firewall allows WebSocket connections
3. Check for proxy interference
4. Monitor browser console for connection errors

### Slow Performance
1. Check network latency
2. Verify caching is working
3. Monitor API response times
4. Check for rate limiting

## Next Steps

1. **Add more real-time features**
   - Order book updates
   - News feed integration
   - Multi-symbol streaming

2. **Enhance error handling**
   - User-friendly error messages
   - Automatic retry logic
   - Offline mode support

3. **Performance optimization**
   - Data virtualization for large datasets
   - Incremental updates
   - Binary WebSocket format

## Benefits

✅ **Live Market Data**: Real-time prices and volumes
✅ **AI Integration**: Live trading signals from 30+ agents
✅ **Reliable**: Automatic fallbacks and reconnection
✅ **Performant**: Smart caching and efficient updates
✅ **Extensible**: Easy to add new data sources
