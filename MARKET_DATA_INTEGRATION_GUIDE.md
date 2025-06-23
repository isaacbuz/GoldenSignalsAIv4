# Market Data Manager Integration Guide

## Quick Start

1. **Install Dependencies**
```bash
pip install requests-cache
```

2. **Update Your Backend**

Replace all direct `yf.Ticker()` calls with the new market data manager:

```python
# OLD CODE:
ticker = yf.Ticker(symbol)
info = ticker.info
price = info.get('regularMarketPrice', 0)

# NEW CODE:
from src.services.market_data_manager import get_market_data_manager

market_data_manager = get_market_data_manager()
data = await market_data_manager.get_market_data(symbol)
price = data['price']
```

3. **Example Integration in `standalone_backend_optimized.py`**

```python
# At the top of the file
from src.services.market_data_manager import get_market_data_manager

# Initialize once
market_data_manager = get_market_data_manager()

# Replace the get_market_data_cached function:
async def get_market_data_cached(symbol: str) -> Optional[MarketData]:
    """Get market data using the new manager"""
    try:
        data = await market_data_manager.get_market_data(symbol)
        
        return MarketData(
            symbol=data['symbol'],
            price=data['price'],
            change=0,  # Calculate from historical if needed
            change_percent=0,
            volume=data.get('volume', 0),
            timestamp=data['timestamp'].isoformat(),
            bid=None,
            ask=None,
            high=data.get('high', data['price']),
            low=data.get('low', data['price']),
            open=data['price']
        )
    except Exception as e:
        logger.error(f"Market data error for {symbol}: {e}")
        # The manager already handles fallbacks, so this is a critical error
        raise HTTPException(status_code=503, detail=f"Market data unavailable for {symbol}")

# Replace the get_historical_data_cached function:
async def get_historical_data_cached(symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Get historical data using the new manager"""
    try:
        return await market_data_manager.get_historical_data(symbol, period, interval)
    except Exception as e:
        logger.error(f"Historical data error for {symbol}: {e}")
        return None
```

## Environment Variables

Add these to your `.env` file for additional data providers:

```bash
# Optional: Alpha Vantage API Key (free tier available)
ALPHA_VANTAGE_KEY=your_key_here

# Optional: Polygon.io API Key
POLYGON_KEY=your_key_here

# Optional: Finnhub API Key
FINNHUB_KEY=your_key_here
```

## Benefits

1. **No More 401 Errors**: Multiple fallback providers ensure data availability
2. **Better Performance**: Built-in caching and rate limiting
3. **Circuit Breakers**: Automatic recovery from provider failures
4. **Mock Data**: System continues working even without real data
5. **Easy Testing**: Mock provider allows offline development

## Monitoring

The manager logs all provider usage:

```
INFO: Initialized 3 data providers
INFO: Got data for AAPL from YFinanceProvider
WARNING: Circuit breaker opened after 5 failures
WARNING: Using fallback data for TSLA
INFO: Got data for TSLA from AlphaVantageProvider
```

## Advanced Usage

### Custom Providers

Add your own data provider:

```python
from src.services.market_data_manager import DataProvider

class PolygonProvider(DataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def fetch_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Implement Polygon.io API call
        pass
        
    async def fetch_historical(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        # Implement historical data fetch
        pass
```

### Cache Control

```python
# Force fresh data (bypass cache)
data = await market_data_manager.get_market_data(symbol, use_cache=False)

# Check circuit breaker status
for provider_name, breaker in market_data_manager.circuit_breakers.items():
    print(f"{provider_name}: {breaker.state}")
```

## Migration Checklist

- [ ] Install requests-cache dependency
- [ ] Add market_data_manager.py to your project
- [ ] Update backend to use the manager
- [ ] Add environment variables for additional providers
- [ ] Test with both real and mock data
- [ ] Monitor logs for provider performance

## Next Steps

1. **Add More Providers**: Implement Polygon.io, Finnhub, IEX Cloud
2. **WebSocket Support**: Add real-time data streaming
3. **Historical Data Cache**: Store in database for faster access
4. **Provider Health Dashboard**: Monitor provider uptime and performance
5. **Smart Provider Selection**: Choose provider based on symbol type (stocks vs crypto) 