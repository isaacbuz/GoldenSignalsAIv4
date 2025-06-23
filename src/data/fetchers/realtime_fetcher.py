import pandas as pd

async def fetch_realtime_data(symbol):
    return pd.DataFrame({
        "close": [280.0],
        "timestamp": [pd.Timestamp.now()]
    })
