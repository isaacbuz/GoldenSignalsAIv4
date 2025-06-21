# GoldenSignalsAI V2 - Live Data Quick Start Guide

## 🚀 Getting Started with Live Data

This guide will help you connect GoldenSignalsAI to real market data and start receiving trading signals within minutes.

## 📋 Prerequisites

```bash
# Install required packages
pip install yfinance websocket-client aiohttp pandas numpy

# Optional (for professional data sources)
pip install polygon-api-client alpaca-py
```

## 🎯 Quick Start (Free Data)

### 1. Run with Yahoo Finance (No API Key Required)

```bash
# Start live data feed with Yahoo Finance
python start_live_data.py
```

This will:
- Connect to Yahoo Finance (free, no API key needed)
- Stream real-time quotes for AAPL, GOOGL, TSLA, SPY, QQQ, NVDA, META
- Generate AI trading signals every 30 seconds
- Display signals with confidence scores

### 2. What You'll See

```
📊 AAPL: $185.23 (Bid: $185.22, Ask: $185.24, Volume: 45,234,567)
📈 AAPL Options: 50 calls, 50 puts updated

============================================================
🎯 TRADING SIGNALS FOR AAPL
============================================================
✅ ACTION: BUY (Confidence: 78.5%)

📊 Market Sentiment:
   Independent: bullish
   Collaborative: strong_bullish
   Confidence: 82.3%

🤖 Top Contributing Agents:
   1. HybridRSI: buy (85.2%)
   2. HybridVolume: buy (79.8%)
   3. HybridMACD: buy (72.1%)
============================================================
```

## 💎 Professional Data Sources

### 1. Polygon.io Setup (Recommended)

```bash
# Get free API key from https://polygon.io
export POLYGON_API_KEY='your_polygon_api_key_here'

# Run with Polygon data
python start_live_data.py
```

Benefits:
- Real-time quotes with microsecond timestamps
- Full options chains with Greeks
- Historical data for backtesting
- WebSocket streaming for low latency

### 2. Alpaca Markets Setup

```bash
# Get API keys from https://alpaca.markets
export ALPACA_API_KEY='your_api_key'
export ALPACA_SECRET_KEY='your_secret_key'

# For paper trading (recommended for testing)
export ALPACA_PAPER=true
```

### 3. Interactive Brokers Setup

```bash
# Install IB Gateway or TWS
# Default port: 7497 (paper), 7496 (live)

# Install Python API
pip install ib_insync

# Configure in code
IB_GATEWAY_HOST='127.0.0.1'
IB_GATEWAY_PORT=7497
```

## 🔧 Configuration

### 1. Add More Symbols

Edit `start_live_data.py`:

```python
self.symbols = [
    'AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 
    'NVDA', 'META', 'AMZN', 'MSFT',  # Add more here
    'BTC-USD', 'ETH-USD'  # Crypto symbols
]
```

### 2. Adjust Update Frequency

```python
# In start_live_data.py
await self.data_feed.start_streaming(
    self.symbols, 
    interval=1  # Update every 1 second (was 5)
)

# Signal generation frequency
await asyncio.sleep(15)  # Generate signals every 15 seconds (was 30)
```

### 3. Enable Specific Agents

```python
# In hybrid_orchestrator.py, customize which agents to use
self.agents = {
    'HybridRSI': HybridRSIAgent(),
    'HybridVolume': HybridVolumeAgent(),
    'HybridMACD': HybridMACDAgent(),
    'HybridBollinger': HybridBollingerAgent(),
    'HybridPattern': HybridPatternAgent(),  # Add more
    'OptionsFlow': HybridOptionsFlowAgent(),  # Options-specific
}
```

## 📊 Data Flow Architecture

```
Market Data Sources          Data Processing           Agent System
┌─────────────────┐         ┌─────────────┐         ┌──────────────┐
│  Yahoo Finance  │────┬───▶│  Unified    │────────▶│  Agent Data  │
└─────────────────┘    │    │  Data Feed  │         │     Bus      │
┌─────────────────┐    │    └─────────────┘         └──────┬───────┘
│   Polygon.io    │────┤           │                        │
└─────────────────┘    │           ▼                        ▼
┌─────────────────┐    │    ┌─────────────┐         ┌──────────────┐
│     Alpaca      │────┘    │   Adapter   │         │   Hybrid     │
└─────────────────┘         └─────────────┘         │   Agents     │
                                                     └──────────────┘
                                                            │
                                                            ▼
                                                     ┌──────────────┐
                                                     │   Trading    │
                                                     │   Signals    │
                                                     └──────────────┘
```

## 🔍 Monitoring & Debugging

### 1. Enable Debug Logging

```python
# In start_live_data.py
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. Check Data Quality

```python
# Add data quality checks
class DataQualityMonitor:
    def check_quote(self, data: MarketData):
        if data.bid > data.ask:
            logger.warning(f"Invalid spread for {data.symbol}")
        if data.volume < 0:
            logger.error(f"Negative volume for {data.symbol}")
```

### 3. Monitor Agent Performance

```bash
# View agent performance metrics
tail -f logs/agent_performance.log

# Check for errors
grep ERROR logs/*.log
```

## 🎯 Trading Integration

### 1. Paper Trading First

```python
# Add paper trading tracker
class PaperTradingTracker:
    def __init__(self, initial_balance=100000):
        self.balance = initial_balance
        self.positions = {}
        self.trades = []
        
    def execute_signal(self, signal):
        if signal['action'] == 'BUY':
            # Simulate buy
            pass
```

### 2. Connect to Broker (Future)

```python
# Example broker integration
from brokers.robinhood import RobinhoodClient

broker = RobinhoodClient()
if signal['confidence'] > 0.8:
    broker.place_order(
        symbol=signal['symbol'],
        action=signal['action'],
        quantity=calculate_position_size(signal)
    )
```

## 🚨 Common Issues & Solutions

### Issue: "No data returned from Yahoo Finance"
```bash
# Solution: Market might be closed, use different symbols
# Try crypto which trades 24/7
self.symbols = ['BTC-USD', 'ETH-USD']
```

### Issue: "Rate limit exceeded"
```python
# Solution: Add rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=5, period=60)  # 5 calls per minute
def fetch_data(symbol):
    return source.get_quote(symbol)
```

### Issue: "Import errors"
```bash
# Solution: Ensure you're in the project root
cd /path/to/GoldenSignalsAI_V2
python start_live_data.py
```

## 📈 Next Steps

1. **Test with Paper Money**: Run the system without real money first
2. **Optimize Agents**: Tune agent parameters based on performance
3. **Add More Data**: Integrate news, sentiment, and fundamental data
4. **Backtest Strategies**: Use historical data to validate signals
5. **Deploy Production**: Set up monitoring and alerts for 24/7 operation

## 🔗 Useful Resources

- [Yahoo Finance Symbols](https://finance.yahoo.com/lookup)
- [Polygon.io Docs](https://polygon.io/docs)
- [Alpaca API Guide](https://alpaca.markets/docs)
- [IB API Documentation](https://interactivebrokers.github.io)

## 💡 Pro Tips

1. **Start Simple**: Begin with 2-3 symbols and Yahoo Finance
2. **Monitor Latency**: Track data delays, especially during high volume
3. **Use Caching**: Cache data to reduce API calls
4. **Log Everything**: Comprehensive logging helps debug issues
5. **Test Weekends**: Use crypto symbols to test when markets are closed

---

Ready to start? Run `python start_live_data.py` and watch the AI agents analyze live markets! 