# arbitrage/agents.py
# Defines arbitrage agents for cross-exchange and statistical arbitrage.

import logging
from typing import Dict, List, Optional
import pandas as pd
import time

logger = logging.getLogger(__name__)

class ArbitrageOpportunity:
    def __init__(self, symbol: str, buy_venue: str, sell_venue: str, buy_price: float, sell_price: float, timestamp: float):
        self.symbol = symbol
        self.buy_venue = buy_venue
        self.sell_venue = sell_venue
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.spread = sell_price - buy_price
        self.timestamp = timestamp
        self.status = 'Open'  # Open, Executed, Missed

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'buy_venue': self.buy_venue,
            'sell_venue': self.sell_venue,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'spread': self.spread,
            'timestamp': self.timestamp,
            'status': self.status,
        }

class CrossExchangeArbitrageAgent:
    def __init__(self, data_sources: Dict[str, callable]):
        """
        data_sources: dict mapping venue name to a price-fetching function: (symbol) -> price
        """
        self.data_sources = data_sources
        self.last_opportunities: List[ArbitrageOpportunity] = []

    def find_opportunities(self, symbol: str, min_spread: float = 0.01) -> List[ArbitrageOpportunity]:
        prices = {}
        for venue, fetcher in self.data_sources.items():
            try:
                price = fetcher(symbol)
                if price is not None:
                    prices[venue] = price
            except Exception as e:
                logger.warning(f"Failed to fetch price from {venue} for {symbol}: {e}")
        opps = []
        venues = list(prices.keys())
        for i in range(len(venues)):
            for j in range(len(venues)):
                if i == j:
                    continue
                buy_venue = venues[i]
                sell_venue = venues[j]
                buy_price = prices[buy_venue]
                sell_price = prices[sell_venue]
                spread = sell_price - buy_price
                if spread > min_spread:
                    opp = ArbitrageOpportunity(symbol, buy_venue, sell_venue, buy_price, sell_price, time.time())
                    opps.append(opp)
        self.last_opportunities = opps
        return opps

class StatisticalArbitrageAgent:
    def __init__(self, historical_data_fetcher: callable):
        self.fetcher = historical_data_fetcher

    def detect_mean_reversion(self, symbol: str, window: int = 20, threshold: float = 2.0) -> Optional[Dict]:
        df = self.fetcher(symbol)
        if df is None or len(df) < window:
            return None
        close = df['Close'] if 'Close' in df else df.iloc[:, 0]
        mean = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        zscore = (close - mean) / std
        latest_z = zscore.iloc[-1]
        if abs(latest_z) > threshold:
            return {
                'symbol': symbol,
                'zscore': latest_z,
                'mean': mean.iloc[-1],
                'std': std.iloc[-1],
                'signal': 'Buy' if latest_z < 0 else 'Sell',
                'timestamp': time.time(),
            }
        return None
