"""
Market Data API Endpoints - GoldenSignalsAI V3

REST API endpoints for accessing real-time and historical market data.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime

from ...services.market_data_service import MarketDataService
from ...core.dependencies import get_market_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-data", tags=["market-data"])

# Initialize market data service
market_data_service = MarketDataService()


class MarketDataResponse(BaseModel):
    """Market data response model"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str
    provider: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None


class MarketStatusResponse(BaseModel):
    """Market status response model"""
    is_open: bool
    current_time: str
    market_hours: Dict[str, str]
    next_open: str
    next_close: str


class HistoricalDataPoint(BaseModel):
    """Historical data point model"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@router.get("/{symbol}")
async def get_market_data(symbol: str) -> Dict[str, Any]:
    """Get real-time market data for a symbol"""
    try:
        quote = await market_data_service.get_quote(symbol.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Transform to match frontend expectations
        return {
            "symbol": quote["symbol"],
            "price": quote["price"],
            "change": quote["change"],
            "change_percent": quote["change_percent"],
            "volume": quote["volume"],
            "timestamp": int(datetime.utcnow().timestamp() * 1000)  # milliseconds
        }
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/quote")
async def get_quote(symbol: str) -> Dict[str, Any]:
    """Get detailed quote data for a symbol"""
    try:
        quote = await market_data_service.get_quote(symbol.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        return quote
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/historical")
async def get_historical_data(
    symbol: str,
    period: str = Query("1d", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    interval: str = Query("5m", description="Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
) -> Dict[str, Any]:
    """Get historical price data for a symbol"""
    try:
        hist_data = await market_data_service.get_historical_data(
            symbol.upper(), 
            period=period, 
            interval=interval
        )
        
        if hist_data is None or hist_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for symbol {symbol}")
        
        # Convert DataFrame to list of dictionaries
        data = []
        for index, row in hist_data.iterrows():
            data.append({
                "time": int(index.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quotes")
async def get_multiple_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get quotes for multiple symbols"""
    try:
        quotes = await market_data_service.get_quotes([s.upper() for s in symbols])
        
        # Transform to match frontend expectations
        result = {}
        for symbol, quote in quotes.items():
            if quote:
                result[symbol] = {
                    "symbol": quote["symbol"],
                    "price": quote["price"],
                    "change": quote["change"],
                    "change_percent": quote["change_percent"],
                    "volume": quote["volume"],
                    "timestamp": int(datetime.utcnow().timestamp() * 1000)
                }
        
        return result
    except Exception as e:
        logger.error(f"Error fetching multiple quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/market")
async def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    try:
        status = await market_data_service.get_market_status()
        return status
    except Exception as e:
        logger.error(f"Error fetching market status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/{query}")
async def search_symbols(query: str) -> List[Dict[str, str]]:
    """Search for symbols matching the query"""
    try:
        results = await market_data_service.search_symbols(query)
        return results
    except Exception as e:
        logger.error(f"Error searching symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/options")
async def get_options_chain(symbol: str) -> Dict[str, Any]:
    """Get options chain data for a symbol"""
    try:
        options = await market_data_service.get_options_chain(symbol.upper())
        if not options:
            raise HTTPException(status_code=404, detail=f"No options data found for symbol {symbol}")
        return options
    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/news")
async def get_symbol_news(
    symbol: str,
    limit: int = Query(10, description="Number of news items to return")
) -> List[Dict[str, Any]]:
    """Get latest news for a symbol"""
    try:
        news = await market_data_service.get_news(symbol.upper(), limit=limit)
        return news
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indices/summary")
async def get_indices_summary() -> Dict[str, Any]:
    """Get summary of major market indices"""
    try:
        # Get quotes for major indices
        indices = ["SPY", "QQQ", "DIA", "IWM", "VTI"]
        quotes = await market_data_service.get_quotes(indices)
        
        summary = {}
        for symbol, quote in quotes.items():
            if quote:
                summary[symbol] = {
                    "name": {
                        "SPY": "S&P 500",
                        "QQQ": "NASDAQ 100",
                        "DIA": "Dow Jones",
                        "IWM": "Russell 2000",
                        "VTI": "Total Market"
                    }.get(symbol, symbol),
                    "price": quote["price"],
                    "change": quote["change"],
                    "change_percent": quote["change_percent"],
                    "volume": quote["volume"]
                }
        
        return summary
    except Exception as e:
        logger.error(f"Error fetching indices summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 