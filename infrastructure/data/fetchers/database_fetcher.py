import pandas as pd

async def fetch_stock_data(symbol, timeframe="1d"):
    return pd.DataFrame({
        "open": [100 + i for i in range(100)],
        "high": [105 + i for i in range(100)],
        "low": [95 + i for i in range(100)],
        "close": [100 + i for i in range(100)],
        "volume": [1000000] * 100
    })
