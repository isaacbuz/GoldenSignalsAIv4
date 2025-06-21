#!/usr/bin/env python3
"""
Simple Live Data Fetcher for GoldenSignalsAI
Uses yfinance directly without database dependencies
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SimpleLiveData:
    """Simple live data fetcher using yfinance"""
    
    def __init__(self):
        # API keys (for future use)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
    async def fetch_live_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch live quotes for multiple symbols"""
        quotes = {}
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price from different fields
                price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('price', 0)
                
                quotes[symbol] = {
                    'price': float(price) if price else 0,
                    'change': float(info.get('regularMarketChange', 0)),
                    'changePercent': float(info.get('regularMarketChangePercent', 0)),
                    'volume': int(info.get('regularMarketVolume', 0)),
                    'bid': float(info.get('bid', 0)),
                    'ask': float(info.get('ask', 0)),
                    'high': float(info.get('dayHigh', 0)),
                    'low': float(info.get('dayLow', 0)),
                    'open': float(info.get('regularMarketOpen', 0)),
                    'previousClose': float(info.get('regularMarketPreviousClose', 0)),
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }
                
                logger.info(f"Fetched quote for {symbol}: ${price}")
                
        except Exception as e:
            logger.error(f"Error fetching live quotes: {e}")
        
        return quotes
    
    async def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = '1d'
    ) -> List[Dict]:
        """Fetch historical data from yfinance"""
        try:
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            ticker = yf.Ticker(symbol)
            
            # Convert interval format if needed
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '60m', '60m': '60m', '1d': '1d', '1w': '1wk', '1mo': '1mo'
            }
            yf_interval = interval_map.get(interval, interval)
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return []
            
            # Convert to expected format
            formatted_data = []
            for index, row in data.iterrows():
                formatted_data.append({
                    "time": int(index.timestamp()),
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume']),
                })
            
            logger.info(f"Fetched {len(formatted_data)} data points for {symbol}")
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def initialize(self):
        """Initialize (no-op for simple version)"""
        logger.info("Simple live data fetcher initialized")
        
    async def close(self):
        """Close (no-op for simple version)"""
        pass


# Create singleton instance
simple_live_data = SimpleLiveData() 