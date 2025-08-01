"""
GoldenSignalsAI Market Data MCP Server V2 - Week 2 Implementation
Provides real-time and historical market data access via MCP with rate limiting
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random
import time
from mcp.server import Server
from mcp import types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataMCPV2(Server):
    """MCP server for market data access with rate limiting"""

    def __init__(self):
        super().__init__("goldensignals-market-data")
        logger.info("Initializing MarketDataMCPV2 server...")

        # Cache for market data
        self.cache = {}
        self.cache_ttl = 300  # Cache for 5 minutes

        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Minimum 1 second between requests

        # Mock data mode flag
        self.use_mock_data = False

        # Popular symbols
        self.popular_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ',
            'AMZN', 'META', 'NFLX', 'AMD', 'INTC', 'BABA', 'JPM',
            'V', 'MA', 'BAC', 'WMT', 'DIS', 'PYPL'
        ]

        # Mock company data
        self.mock_companies = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
            'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services'},
            'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software'},
            'TSLA': {'name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'},
            'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors'},
            'AMZN': {'name': 'Amazon.com, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail'},
            'META': {'name': 'Meta Platforms, Inc.', 'sector': 'Technology', 'industry': 'Internet Services'},
            'SPY': {'name': 'SPDR S&P 500 ETF', 'sector': 'ETF', 'industry': 'Large Cap Blend'},
            'QQQ': {'name': 'Invesco QQQ Trust', 'sector': 'ETF', 'industry': 'Large Cap Growth'}
        }

        logger.info("MarketDataMCPV2 server initialized successfully")

    async def _check_rate_limit(self, key: str) -> bool:
        """Check if we can make a request (rate limiting)"""
        now = time.time()
        if key in self.last_request_time:
            elapsed = now - self.last_request_time[key]
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)

        self.last_request_time[key] = time.time()
        return True

    async def handle_initialize(self) -> types.InitializeResult:
        """Initialize the MCP server with capabilities"""
        logger.info("Handling MCP initialization...")

        return types.InitializeResult(
            protocol_version="2024-11-05",
            capabilities=types.ServerCapabilities(
                tools=types.ToolsCapability(list_changed=False),
                resources=types.ResourcesCapability(subscribe=True, list_changed=True)
            ),
            server_info=types.Implementation(
                name="GoldenSignals Market Data Server V2",
                version="2.0.0"
            )
        )

    async def handle_list_tools(self) -> List[types.Tool]:
        """List available market data tools"""
        logger.info("Listing available MCP tools...")

        return [
            types.Tool(
                name="get_quote",
                description="Get real-time quote for a stock symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, TSLA)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            types.Tool(
                name="get_historical_data",
                description="Get historical price data for a symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "days": {
                            "type": "integer",
                            "description": "Number of days of history (1-365)",
                            "default": 30
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            types.Tool(
                name="get_market_summary",
                description="Get market summary with major indices and top movers",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="compare_stocks",
                description="Compare multiple stocks side by side",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock symbols to compare"
                        }
                    },
                    "required": ["symbols"]
                }
            ),
            types.Tool(
                name="get_volatility",
                description="Calculate volatility metrics for a stock",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "period_days": {
                            "type": "integer",
                            "description": "Period in days for volatility calculation",
                            "default": 30
                        }
                    },
                    "required": ["symbol"]
                }
            )
        ]

    def _get_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key for a request"""
        return f"{tool_name}:{json.dumps(params, sort_keys=True)}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        return (datetime.now() - cached_time).total_seconds() < self.cache_ttl

    def _generate_mock_price(self, symbol: str, base_price: Optional[float] = None) -> float:
        """Generate a mock stock price"""
        if base_price is None:
            # Generate base prices for known symbols
            base_prices = {
                'AAPL': 190.0, 'GOOGL': 140.0, 'MSFT': 380.0, 'TSLA': 250.0,
                'NVDA': 500.0, 'AMZN': 170.0, 'META': 350.0, 'SPY': 450.0,
                'QQQ': 380.0, 'AMD': 120.0, 'INTC': 45.0, 'NFLX': 450.0
            }
            base_price = base_prices.get(symbol, 100.0)

        # Add some random variation (±5%)
        variation = random.uniform(-0.05, 0.05)
        return round(base_price * (1 + variation), 2)

    async def _get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol (with mock data fallback)"""
        await self._check_rate_limit('quote')

        try:
            # Try to get real data with yfinance
            if not self.use_mock_data:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol.upper())
                    info = ticker.info

                    if info and 'regularMarketPrice' in info:
                        current_price = info.get('regularMarketPrice', 0)
                        previous_close = info.get('regularMarketPreviousClose', info.get('previousClose', 0))

                        if current_price > 0:
                            change = current_price - previous_close if previous_close else 0
                            change_percent = (change / previous_close * 100) if previous_close else 0

                            return {
                                "symbol": symbol.upper(),
                                "name": info.get('longName', self.mock_companies.get(symbol, {}).get('name', symbol)),
                                "price": current_price,
                                "previousClose": previous_close,
                                "change": round(change, 2),
                                "changePercent": round(change_percent, 2),
                                "dayHigh": info.get('dayHigh', current_price * 1.02),
                                "dayLow": info.get('dayLow', current_price * 0.98),
                                "volume": info.get('volume', random.randint(1000000, 50000000)),
                                "avgVolume": info.get('averageVolume', random.randint(1000000, 50000000)),
                                "marketCap": info.get('marketCap', 0),
                                "bid": info.get('bid', current_price - 0.01),
                                "ask": info.get('ask', current_price + 0.01),
                                "timestamp": datetime.now().isoformat(),
                                "dataSource": "live"
                            }
                except Exception as e:
                    logger.warning(f"Failed to get live data for {symbol}: {e}. Using mock data.")
                    self.use_mock_data = True

            # Use mock data
            current_price = self._generate_mock_price(symbol)
            previous_close = self._generate_mock_price(symbol, current_price * 0.99)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100)

            return {
                "symbol": symbol.upper(),
                "name": self.mock_companies.get(symbol, {}).get('name', f'{symbol} Corporation'),
                "price": current_price,
                "previousClose": previous_close,
                "change": round(change, 2),
                "changePercent": round(change_percent, 2),
                "dayHigh": round(current_price * 1.02, 2),
                "dayLow": round(current_price * 0.98, 2),
                "volume": random.randint(1000000, 50000000),
                "avgVolume": random.randint(1000000, 50000000),
                "marketCap": int(current_price * random.randint(100000000, 1000000000)),
                "bid": round(current_price - 0.01, 2),
                "ask": round(current_price + 0.01, 2),
                "timestamp": datetime.now().isoformat(),
                "dataSource": "mock"
            }

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            raise

    async def _get_historical_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get historical price data (with mock data generation)"""
        await self._check_rate_limit('historical')

        try:
            # Generate mock historical data
            data = []
            base_price = self._generate_mock_price(symbol)

            for i in range(days):
                date = datetime.now() - timedelta(days=days-i-1)

                # Generate OHLCV data with some realistic patterns
                open_price = base_price * (1 + random.uniform(-0.02, 0.02))
                close_price = open_price * (1 + random.uniform(-0.03, 0.03))
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
                volume = random.randint(1000000, 50000000)

                data.append({
                    "date": date.date().isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume
                })

                # Update base price for next day
                base_price = close_price

            # Calculate statistics
            prices = [d['close'] for d in data]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

            return {
                "symbol": symbol.upper(),
                "period": f"{days} days",
                "dataPoints": len(data),
                "data": data,
                "statistics": {
                    "highestPrice": round(max(prices), 2),
                    "lowestPrice": round(min(prices), 2),
                    "averagePrice": round(sum(prices) / len(prices), 2),
                    "volatility": round(self._calculate_volatility(returns) * 100, 2),
                    "totalReturn": round((prices[-1] / prices[0] - 1) * 100, 2) if prices else 0
                },
                "timestamp": datetime.now().isoformat(),
                "dataSource": "mock"
            }

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise

    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate volatility from returns"""
        if not returns:
            return 0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    async def _get_market_summary(self) -> Dict[str, Any]:
        """Get market summary with indices and movers"""
        await self._check_rate_limit('market_summary')

        try:
            # Major indices
            indices = [
                {'symbol': '^GSPC', 'name': 'S&P 500'},
                {'symbol': '^DJI', 'name': 'Dow Jones'},
                {'symbol': '^IXIC', 'name': 'NASDAQ'},
                {'symbol': '^RUT', 'name': 'Russell 2000'},
                {'symbol': '^VIX', 'name': 'VIX'}
            ]

            index_data = []
            for idx in indices:
                quote = await self._get_quote(idx['symbol'])
                index_data.append({
                    "symbol": idx['symbol'],
                    "name": idx['name'],
                    "price": quote['price'],
                    "change": quote['change'],
                    "changePercent": quote['changePercent']
                })

            # Get data for popular stocks
            stock_data = []
            for symbol in self.popular_symbols[:10]:
                quote = await self._get_quote(symbol)
                stock_data.append({
                    "symbol": symbol,
                    "name": quote['name'],
                    "price": quote['price'],
                    "change": quote['change'],
                    "changePercent": quote['changePercent'],
                    "volume": quote['volume']
                })

            # Sort by change percentage
            stock_data.sort(key=lambda x: x['changePercent'], reverse=True)

            # Determine market status
            now = datetime.now()
            is_weekday = now.weekday() < 5
            market_hours = 9 <= now.hour < 16
            market_status = "open" if is_weekday and market_hours else "closed"

            return {
                "marketStatus": market_status,
                "timestamp": now.isoformat(),
                "indices": index_data,
                "topGainers": stock_data[:3],
                "topLosers": stock_data[-3:],
                "popularStocks": stock_data,
                "dataSource": "mock"
            }

        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            raise

    async def _compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        await self._check_rate_limit('compare')

        try:
            comparisons = []

            for symbol in symbols[:5]:  # Limit to 5 symbols
                quote = await self._get_quote(symbol)
                hist = await self._get_historical_data(symbol, 30)

                comparisons.append({
                    "symbol": symbol,
                    "name": quote['name'],
                    "price": quote['price'],
                    "change": quote['change'],
                    "changePercent": quote['changePercent'],
                    "volume": quote['volume'],
                    "marketCap": quote['marketCap'],
                    "monthlyReturn": hist['statistics']['totalReturn'],
                    "volatility": hist['statistics']['volatility'],
                    "avgPrice30d": hist['statistics']['averagePrice']
                })

            # Sort by performance
            comparisons.sort(key=lambda x: x['changePercent'], reverse=True)

            return {
                "symbols": symbols,
                "comparisons": comparisons,
                "bestPerformer": comparisons[0]['symbol'] if comparisons else None,
                "worstPerformer": comparisons[-1]['symbol'] if comparisons else None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error comparing stocks: {e}")
            raise

    async def _get_volatility(self, symbol: str, period_days: int = 30) -> Dict[str, Any]:
        """Calculate detailed volatility metrics"""
        await self._check_rate_limit('volatility')

        try:
            # Get historical data
            hist = await self._get_historical_data(symbol, period_days)
            prices = [d['close'] for d in hist['data']]

            if len(prices) < 2:
                raise ValueError("Not enough data to calculate volatility")

            # Calculate various volatility metrics
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

            # Daily volatility
            daily_vol = self._calculate_volatility(returns)

            # Annualized volatility (assuming 252 trading days)
            annual_vol = daily_vol * (252 ** 0.5)

            # Calculate high-low volatility (Parkinson)
            hl_vol = 0
            for d in hist['data']:
                if d['high'] > 0 and d['low'] > 0:
                    hl_vol += (math.log(d['high'] / d['low'])) ** 2
            hl_vol = (hl_vol / len(hist['data']) / (4 * math.log(2))) ** 0.5 * (252 ** 0.5) if hist['data'] else 0

            # Find largest moves
            daily_moves = sorted(
                [(hist['data'][i]['date'], returns[i-1] * 100) for i in range(1, len(hist['data']))],
                key=lambda x: abs(x[1]),
                reverse=True
            )

            return {
                "symbol": symbol,
                "period": f"{period_days} days",
                "metrics": {
                    "dailyVolatility": round(daily_vol * 100, 2),
                    "annualizedVolatility": round(annual_vol * 100, 2),
                    "highLowVolatility": round(hl_vol * 100, 2),
                    "averageReturn": round(sum(returns) / len(returns) * 100, 4) if returns else 0,
                    "maxDailyGain": round(max(returns) * 100, 2) if returns else 0,
                    "maxDailyLoss": round(min(returns) * 100, 2) if returns else 0
                },
                "largestMoves": [
                    {"date": move[0], "changePercent": round(move[1], 2)}
                    for move in daily_moves[:5]
                ],
                "riskLevel": "High" if annual_vol > 0.4 else "Medium" if annual_vol > 0.2 else "Low",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            raise

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute tool calls"""
        logger.info(f"Executing tool: {name} with arguments: {arguments}")

        try:
            # Check cache first
            cache_key = self._get_cache_key(name, arguments)
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                logger.info(f"Returning cached result for {name}")
                result = self.cache[cache_key]['data']
            else:
                # Execute the appropriate function
                if name == "get_quote":
                    result = await self._get_quote(arguments["symbol"])

                elif name == "get_historical_data":
                    result = await self._get_historical_data(
                        arguments["symbol"],
                        arguments.get("days", 30)
                    )

                elif name == "get_market_summary":
                    result = await self._get_market_summary()

                elif name == "compare_stocks":
                    result = await self._compare_stocks(arguments["symbols"])

                elif name == "get_volatility":
                    result = await self._get_volatility(
                        arguments["symbol"],
                        arguments.get("period_days", 30)
                    )

                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Cache the result
                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                }

            # Return formatted result
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            error_result = {
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }
            return [types.TextContent(
                type="text",
                text=json.dumps(error_result, indent=2)
            )]

    async def handle_list_resources(self) -> List[types.Resource]:
        """List available resources for subscription"""
        logger.info("Listing available MCP resources...")

        return [
            types.Resource(
                uri="market://quotes/stream",
                name="Real-time Quotes Stream",
                description="Stream of real-time quotes for popular stocks",
                mime_type="application/json"
            ),
            types.Resource(
                uri="market://watchlist/default",
                name="Default Watchlist",
                description="Pre-configured watchlist with popular stocks",
                mime_type="application/json"
            )
        ]

    async def handle_read_resource(self, uri: str) -> str:
        """Read resource data"""
        logger.info(f"Reading resource: {uri}")

        try:
            if uri == "market://quotes/stream":
                # Get quotes for top stocks
                quotes = []
                for symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']:
                    quote = await self._get_quote(symbol)
                    quotes.append({
                        "symbol": symbol,
                        "price": quote['price'],
                        "change": quote['changePercent'],
                        "volume": quote['volume']
                    })

                result = {
                    "type": "quote_stream",
                    "quotes": quotes,
                    "timestamp": datetime.now().isoformat()
                }

            elif uri == "market://watchlist/default":
                # Return default watchlist
                result = {
                    "type": "watchlist",
                    "name": "Default Tech Watchlist",
                    "symbols": ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN'],
                    "timestamp": datetime.now().isoformat()
                }

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return json.dumps({"error": str(e), "uri": uri})

# Add math import for volatility calculations
import math

# Run the server
if __name__ == "__main__":
    import mcp.server.stdio

    async def main():
        logger.info("Starting GoldenSignals Market Data MCP Server V2...")

        try:
            server = MarketDataMCPV2()

            # Run the MCP server
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.handle_initialize,
                    server.handle_list_tools,
                    server.handle_call_tool,
                    server.handle_list_resources,
                    server.handle_read_resource
                )
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise

    # Run the async main function
    asyncio.run(main())
