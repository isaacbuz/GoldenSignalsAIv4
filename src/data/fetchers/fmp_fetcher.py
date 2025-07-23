import os

import pandas as pd
import requests

FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("REACT_APP_FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

def fetch_fmp_ohlcv(symbol, timeframe="1d", limit=100):
    """
    Fetch historical OHLCV data for a symbol from Financial Modeling Prep.
    timeframe: '1min', '5min', '15min', '30min', '1hour', '4hour', '1d'
    Returns: pd.DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
    """
    endpoint = f"/historical-chart/{timeframe}/{symbol.upper()}"
    url = f"{FMP_BASE_URL}{endpoint}?apikey={FMP_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    if not data or not isinstance(data, list):
        raise ValueError(f"No data returned from FMP for {symbol}")
    df = pd.DataFrame(data)
    # FMP returns newest first; reverse for chronological
    df = df.iloc[::-1].reset_index(drop=True)
    if limit:
        df = df.tail(limit)
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def fetch_fmp_quote(symbol):
    """
    Fetch latest quote for a symbol from FMP.
    Returns: dict with price info
    """
    url = f"{FMP_BASE_URL}/quote/{symbol.upper()}?apikey={FMP_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    if not data or not isinstance(data, list):
        raise ValueError(f"No quote data from FMP for {symbol}")
    return data[0]
