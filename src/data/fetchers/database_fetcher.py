import pandas as pd

from .fmp_fetcher import fetch_fmp_ohlcv


async def fetch_stock_data(symbol, timeframe="1d", source="mock"):
    """
    Fetch OHLCV data for a symbol from the specified source.
    source: 'mock' (default), 'fmp' (Financial Modeling Prep)
    """
    if source == "fmp":
        # FMP is synchronous, so run in thread if called from async context
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_fmp_ohlcv, symbol, timeframe)
    # Default: mock data
    import pandas as pd

    return pd.DataFrame(
        {
            "open": [100 + i for i in range(100)],
            "high": [105 + i for i in range(100)],
            "low": [95 + i for i in range(100)],
            "close": [100 + i for i in range(100)],
            "volume": [1000000] * 100,
        }
    )
