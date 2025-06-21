# After-Hours Data Handling in GoldenSignalsAI V3

## Overview

GoldenSignalsAI V3 includes intelligent after-hours data handling to ensure 24/7 availability for analysis, backtesting, and signal generation even when markets are closed.

## Key Features

### 1. **Smart Error Detection**
The system distinguishes between different types of data unavailability:
- **Market Closed**: Trading hours have ended
- **Invalid Symbol**: Symbol doesn't exist or is delisted
- **Network Error**: Connection issues
- **API Limit**: Rate limiting from data provider
- **No Data**: Other data availability issues

### 2. **Automatic Market Hours Detection**
- Checks US market hours (9:30 AM - 4:00 PM ET)
- Considers weekdays vs weekends
- Calculates next market open time
- Provides clear status messages

### 3. **Intelligent Caching System**
- Caches successful data fetches during market hours
- Stores both tick data and historical data
- 24-hour cache TTL (Time To Live)
- Both memory and disk persistence

### 4. **Graceful Fallback**
When markets are closed:
1. First attempts to fetch data (some providers have delayed data)
2. If unavailable, checks if it's due to market hours
3. Loads cached data if available
4. Returns data with appropriate warnings

## API Response Examples

### During Market Hours
```json
{
  "symbol": "AAPL",
  "price": 175.23,
  "volume": 45123456,
  "data_source": "live",
  "timestamp": "2024-01-10T14:30:00"
}
```

### After Market Hours (with cache)
```json
{
  "symbol": "AAPL",
  "price": 175.23,
  "volume": 45123456,
  "data_source": "cache",
  "warning": "Market is closed. After-hours trading. Next open: 2024-01-11T09:30:00",
  "timestamp": "2024-01-10T16:00:00"
}
```

### Signal Generation After Hours
```json
{
  "symbol": "AAPL",
  "signal": "BUY",
  "confidence": 0.72,
  "is_after_hours": true,
  "market_status": {
    "is_open": false,
    "reason": "After-hours trading",
    "next_open": "2024-01-11T09:30:00"
  },
  "indicators": {
    "rsi": 45.2,
    "macd": 1.23,
    "after_hours_data": true,
    "market_status": "After-hours trading"
  }
}
```

## Error Handling Examples

### Market Closed Error
```json
{
  "detail": {
    "message": "Market is closed. Weekend - market closed. Next open: 2024-01-15T09:30:00",
    "reason": "market_closed",
    "is_recoverable": true,
    "suggested_action": "Use cached data or wait for market open",
    "timestamp": "2024-01-13T10:00:00"
  }
}
```

### Invalid Symbol Error
```json
{
  "detail": {
    "message": "Symbol INVALID may be invalid or delisted",
    "reason": "invalid_symbol",
    "is_recoverable": false,
    "suggested_action": "Verify symbol is correct and actively traded",
    "timestamp": "2024-01-10T14:30:00"
  }
}
```

## Implementation Details

### Market Hours Check
```python
def check_market_hours(self) -> MarketHours:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    market_open = time(9, 30)
    market_close = time(16, 0)
    is_weekday = now.weekday() < 5
    is_within_hours = market_open <= now.time() <= market_close
    is_open = is_weekday and is_within_hours
```

### Error Detection
```python
def detect_error_reason(self, symbol: str, error: Exception) -> MarketDataError:
    # Analyzes error message and market hours
    # Returns appropriate error classification
    # Suggests recovery actions
```

### Cache Usage
```python
# During data fetch
if hist.empty and not market_hours.is_open:
    cached_tick = self.cache.get_tick(symbol)
    if cached_tick:
        return cached_tick, None
```

## Testing

Run the test script to see after-hours handling in action:

```bash
python test_after_hours.py
```

The test will:
1. Check current market status
2. Attempt to fetch data for multiple symbols
3. Show cache behavior
4. Demonstrate signal generation with after-hours data

## Benefits

1. **24/7 Availability**: Users can access the system anytime
2. **Backtesting**: Historical analysis works even when markets are closed
3. **Planning**: Users can prepare strategies outside market hours
4. **Global Users**: Accommodates users in different time zones
5. **Reliability**: Graceful degradation instead of errors

## Configuration

Cache settings can be adjusted in `MarketDataCache`:
- `cache_dir`: Directory for persistent cache (default: "data/market_cache")
- `cache_ttl`: Cache time-to-live in seconds (default: 86400 = 24 hours)

## Future Enhancements

1. **Extended Hours Data**: Support pre-market and after-hours trading data
2. **Multi-Market Support**: Handle different market hours for international exchanges
3. **Smart Cache Warming**: Pre-fetch data before market close
4. **Historical Gap Filling**: Automatically fill missing data points
5. **Cache Analytics**: Track cache hit rates and optimize performance 