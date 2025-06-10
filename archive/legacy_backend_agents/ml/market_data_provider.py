import yfinance as yf
from typing import Dict

class MarketDataProvider:
    """
    Fetches real-time and historical market data for a given symbol using Yahoo Finance.
    Extendable to other APIs (Alpha Vantage, Polygon.io).
    """
    def __init__(self, symbol: str):
        self.symbol = symbol

    def get_latest(self) -> Dict:
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period="1d")
        if data.empty:
            return {}
        row = data.iloc[-1]
        return {
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }

    def get_history(self, period="1mo", interval="1d") -> Dict:
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period, interval=interval)
        return data.to_dict('records')
