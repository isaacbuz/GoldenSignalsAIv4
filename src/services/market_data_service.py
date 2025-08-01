#!/usr/bin/env python3
"""
Market Data Service - Fetches and processes real-time market data

This service provides:
- Real-time quotes and historical data
- Technical indicator calculations
- Options chain data
- Market status and news
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

from src.services.cache_service import DataType, get_cache_service
from src.services.monitoring_service import get_monitoring_service
from src.services.rate_limit_handler import RequestPriority, get_rate_limit_handler
from src.services.websocket_service import get_websocket_service
from src.utils.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


# Constants
DEFAULT_QUOTE_TTL = 300  # 5 minutes
DEFAULT_HISTORICAL_TTL = 600  # 10 minutes
MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests
MAX_CACHE_SIZE = 1000
MAX_HISTORICAL_CACHE_SIZE = 500


# Common symbol list for demo/search
COMMON_SYMBOLS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "META", "name": "Meta Platforms Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "WMT", "name": "Walmart Inc."},
    {"symbol": "PG", "name": "Procter & Gamble Co."},
    {"symbol": "MA", "name": "Mastercard Inc."},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
    {"symbol": "HD", "name": "The Home Depot Inc."},
    {"symbol": "DIS", "name": "The Walt Disney Company"},
    {"symbol": "BAC", "name": "Bank of America Corp."},
    {"symbol": "ADBE", "name": "Adobe Inc."},
    {"symbol": "CRM", "name": "Salesforce Inc."},
    {"symbol": "NFLX", "name": "Netflix Inc."},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF"},
]


class MarketDataService:
    """Service for fetching and processing market data"""

    def __init__(self):
        # Initialize services using dependency injection
        self.rate_limit_handler = get_rate_limit_handler()
        self.websocket_service = get_websocket_service()
        self.cache_service = get_cache_service()
        self.monitoring_service = get_monitoring_service()

        # Internal caches with configurable TTL
        self.quote_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=DEFAULT_QUOTE_TTL)
        self.historical_cache = TTLCache(
            maxsize=MAX_HISTORICAL_CACHE_SIZE, ttl=DEFAULT_HISTORICAL_TTL
        )

        # Rate limiting configuration
        self.last_request_time: Dict[str, float] = {}
        self.min_request_interval = MIN_REQUEST_INTERVAL

        # WebSocket lazy initialization flag
        self._websocket_initialized = False

    async def _ensure_websocket(self) -> None:
        """Ensure WebSocket connection is established"""
        if not self._websocket_initialized:
            self._websocket_initialized = True
            await self.websocket_service.connect()

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol"""
        await self._ensure_websocket()

        try:
            # Use cache service with automatic fallback
            quote_data = await self.cache_service.get(
                DataType.QUOTE, symbol, fetch_func=lambda: self._fetch_quote(symbol)
            )

            # Log metrics
            if quote_data:
                logger.debug(f"Quote for {symbol}: ${quote_data.get('price', 0):.2f}")

            return quote_data

        except Exception as e:
            self._handle_error("get_quote", e, {"symbol": symbol})
            return None

    async def _fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote using rate limit handler"""
        return await self.rate_limit_handler.get_quote(symbol, priority=RequestPriority.NORMAL)

    async def subscribe_to_symbol(
        self, symbol: str, callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """Subscribe to real-time updates for a symbol"""
        await self._ensure_websocket()
        return await self.websocket_service.subscribe([symbol], callback=callback)

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols with batch processing"""
        try:
            # Batch fetch with rate limiting
            quotes = await self.rate_limit_handler.batch_get_quotes(
                symbols, priority=RequestPriority.NORMAL
            )

            # Cache successful results
            for symbol, quote in quotes.items():
                if quote:
                    await self.cache_service.set(DataType.QUOTE, symbol, quote)

            return quotes

        except Exception as e:
            self._handle_error("get_quotes", e, {"count": len(symbols)})
            return {}

    async def get_historical_data(
        self, symbol: str, period: str = "1d", interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol"""
        cache_key = f"{symbol}_{period}_{interval}"

        try:
            # Fetch with caching
            hist_data = await self.cache_service.get(
                DataType.HISTORICAL,
                cache_key,
                fetch_func=lambda: self._fetch_historical(symbol, period, interval),
            )

            if hist_data is not None and not hist_data.empty:
                # Add technical indicators
                hist_data = self._add_technical_indicators(hist_data)
                logger.debug(f"Retrieved {len(hist_data)} records for {symbol}")

            return hist_data

        except Exception as e:
            self._handle_error("get_historical_data", e, {"symbol": symbol})
            return None

    async def _fetch_historical(
        self, symbol: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data using rate limit handler"""
        return await self.rate_limit_handler.get_historical_data(
            symbol, period=period, interval=interval, priority=RequestPriority.NORMAL
        )

    def _handle_error(self, operation: str, error: Exception, context: Dict[str, Any]) -> None:
        """Centralized error handling"""
        self.monitoring_service.record_error(operation, context)
        logger.error(f"{operation} failed: {str(error)}", extra=context)

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        if df.empty or len(df) < 20:
            return df

        try:
            # Convert DataFrame to list of dicts for TechnicalIndicators
            data = df.to_dict("records")

            # Ensure column names match expected format
            formatted_data = []
            for row in data:
                formatted_data.append(
                    {
                        "open": row.get("Open", 0),
                        "high": row.get("High", 0),
                        "low": row.get("Low", 0),
                        "close": row.get("Close", 0),
                        "volume": row.get("Volume", 0),
                    }
                )

            # Calculate indicators
            indicators = TechnicalIndicators.calculate_all_indicators(formatted_data)

            # Add scalar indicators to DataFrame
            for key, value in indicators.items():
                if isinstance(value, (int, float)):
                    df[key] = value
                elif isinstance(value, list) and key in ["support_levels", "resistance_levels"]:
                    # Store as string representation for now
                    df[key] = str(value)

            return df

        except Exception as e:
            logger.warning(f"Could not add indicators: {e}")
            return df

    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and major indices"""
        try:
            now = datetime.now(timezone.utc)

            # NYSE market hours (EST/EDT)
            market_open = now.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM EST
            market_close = now.replace(hour=21, minute=0, second=0, microsecond=0)  # 4:00 PM EST

            is_weekday = now.weekday() < 5
            is_market_hours = market_open <= now <= market_close
            is_open = is_weekday and is_market_hours

            # Fetch major indices
            indices = await self.get_quotes(["SPY", "QQQ", "DIA", "IWM"])

            return {
                "is_open": is_open,
                "status": "open" if is_open else "closed",
                "current_time": now.isoformat(),
                "market_open": market_open.isoformat(),
                "market_close": market_close.isoformat(),
                "indices": indices,
            }

        except Exception as e:
            self._handle_error("get_market_status", e, {})
            return {"is_open": False, "status": "error"}

    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for symbols matching the query"""
        if not query:
            return []

        try:
            query_lower = query.lower()

            # Filter symbols by query
            results = [
                symbol_info
                for symbol_info in COMMON_SYMBOLS
                if query_lower in symbol_info["symbol"].lower()
                or query_lower in symbol_info["name"].lower()
            ]

            return results[:limit]

        except Exception as e:
            self._handle_error("search_symbols", e, {"query": query})
            return []

    async def get_options_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain data for a symbol"""
        try:
            # Apply rate limiting
            await self._apply_rate_limit(symbol)

            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return None

            # Get nearest expiration options
            nearest_expiry = expirations[0]
            option_chain = ticker.option_chain(nearest_expiry)

            return {
                "symbol": symbol,
                "expirations": list(expirations),
                "selected_expiry": nearest_expiry,
                "calls": option_chain.calls.head(20).to_dict("records"),
                "puts": option_chain.puts.head(20).to_dict("records"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self._handle_error("get_options_chain", e, {"symbol": symbol})
            return None

    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest news for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news[:limit]

            return [self._format_news_item(item) for item in news_items]

        except Exception as e:
            self._handle_error("get_news", e, {"symbol": symbol})
            return []

    def _format_news_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single news item"""
        return {
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", ""),
            "published": datetime.fromtimestamp(
                item.get("providerPublishTime", 0), tz=timezone.utc
            ).isoformat(),
            "type": item.get("type", ""),
            "thumbnail": self._extract_thumbnail(item),
        }

    def _extract_thumbnail(self, item: Dict[str, Any]) -> str:
        """Extract thumbnail URL from news item"""
        thumbnail = item.get("thumbnail")
        if not thumbnail:
            return ""

        resolutions = thumbnail.get("resolutions", [])
        return resolutions[0].get("url", "") if resolutions else ""

    async def _apply_rate_limit(self, symbol: str) -> None:
        """Apply rate limiting for a symbol"""
        import time

        current_time = time.time()
        last_request = self.last_request_time.get(symbol, 0)

        time_since_last = current_time - last_request
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time[symbol] = time.time()

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return self.monitoring_service.get_dashboard_data()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache_service.get_stats()
