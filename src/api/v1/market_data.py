"""
Market Data API Endpoints - GoldenSignalsAI V3

REST API endpoints for accessing real-time and historical market data.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from ...services.market_data_service import MarketDataService
from ...core.dependencies import get_market_data_service

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.get("/{symbol}", response_model=MarketDataResponse)
async def get_market_data(
    symbol: str,
    market_service: MarketDataService = Depends(get_market_data_service)
) -> MarketDataResponse:
    """
    Get current market data for a specific symbol.
    """
    try:
        data = await market_service.get_current_price(symbol.upper())
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Market data not found for symbol '{symbol}'"
            )
        
        return MarketDataResponse(**data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve market data for '{symbol}'"
        )


@router.post("/batch", response_model=Dict[str, MarketDataResponse])
async def get_multiple_market_data(
    symbols: List[str],
    market_service: MarketDataService = Depends(get_market_data_service)
) -> Dict[str, MarketDataResponse]:
    """
    Get current market data for multiple symbols.
    """
    try:
        if len(symbols) > 50:  # Limit batch requests
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot request more than 50 symbols at once"
            )
        
        data = await market_service.get_multiple_quotes([s.upper() for s in symbols])
        
        result = {}
        for symbol, market_data in data.items():
            if market_data:
                result[symbol] = MarketDataResponse(**market_data)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch market data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batch market data"
        )


@router.get("/{symbol}/historical", response_model=List[HistoricalDataPoint])
async def get_historical_data(
    symbol: str,
    period: str = Query("1d", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("5m", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    market_service: MarketDataService = Depends(get_market_data_service)
) -> List[HistoricalDataPoint]:
    """
    Get historical market data for a specific symbol.
    """
    try:
        data = await market_service.get_historical_data(
            symbol.upper(), 
            period=period,
            interval=interval
        )
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Historical data not found for symbol '{symbol}'"
            )
        
        # Convert to response format
        historical_data = []
        for point in data:
            historical_data.append(HistoricalDataPoint(
                timestamp=point.get("timestamp", ""),
                open=point.get("open", 0.0),
                high=point.get("high", 0.0),
                low=point.get("low", 0.0),
                close=point.get("close", 0.0),
                volume=point.get("volume", 0)
            ))
        
        return historical_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve historical data for '{symbol}'"
        )


@router.get("/status", response_model=MarketStatusResponse)
async def get_market_status(
    market_service: MarketDataService = Depends(get_market_data_service)
) -> MarketStatusResponse:
    """
    Get current market status and trading hours.
    """
    try:
        status_data = await market_service.get_market_status()
        
        return MarketStatusResponse(
            is_open=status_data.get("is_open", False),
            current_time=status_data.get("current_time", ""),
            market_hours=status_data.get("market_hours", {}),
            next_open=status_data.get("next_open", ""),
            next_close=status_data.get("next_close", "")
        )
        
    except Exception as e:
        logger.error(f"Error getting market status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve market status"
        )


@router.get("/{symbol}/options")
async def get_options_data(
    symbol: str,
    market_service: MarketDataService = Depends(get_market_data_service)
) -> Dict[str, Any]:
    """
    Get options data for a specific symbol.
    """
    try:
        options_data = await market_service.get_options_data(symbol.upper())
        
        if not options_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Options data not found for symbol '{symbol}'"
            )
        
        return options_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting options data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve options data for '{symbol}'"
        ) 