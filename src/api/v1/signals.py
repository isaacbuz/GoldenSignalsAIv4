"""
Trading Signals API Endpoints - GoldenSignalsAI V3

Endpoints for retrieving, creating, and managing trading signals.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.signal_service import SignalService
from src.core.dependencies import get_signal_service, get_current_user
from src.models.signals import Signal, SignalType, SignalStrength
from src.core.database import get_db
from src.models.schemas import SignalCreate

router = APIRouter()


class SignalResponse(BaseModel):
    """Response model for signal data"""
    signal_id: str
    symbol: str
    signal_type: str
    confidence: float
    strength: str
    source: str
    current_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_score: Optional[float]
    reasoning: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]


class SignalListResponse(BaseModel):
    """Response model for signal lists"""
    signals: List[SignalResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class SignalAnalyticsResponse(BaseModel):
    """Response model for signal analytics"""
    total_signals: int
    executed_signals: int
    profitable_signals: int
    win_rate: float
    avg_confidence: float
    avg_return: float
    signals_by_source: dict
    signals_by_type: dict


@router.get("/{symbol}", response_model=SignalListResponse)
async def get_signals_for_symbol(
    symbol: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    source: Optional[str] = Query(None, description="Filter by signal source"),
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    hours_back: Optional[int] = Query(24, ge=1, le=168, description="Hours to look back"),
    signal_service: SignalService = Depends(get_signal_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get trading signals for a specific symbol with filtering options.
    
    - **symbol**: Stock symbol (e.g., AAPL, TSLA)
    - **page**: Page number for pagination
    - **page_size**: Number of signals per page
    - **source**: Filter by agent source (optional)
    - **signal_type**: Filter by signal type (BUY, SELL, HOLD)
    - **min_confidence**: Minimum confidence threshold
    - **hours_back**: How many hours back to search
    """
    try:
        since = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Get signals from service
        signals, total = await signal_service.get_signals_paginated(
            symbol=symbol.upper(),
            source=source,
            signal_type=signal_type,
            min_confidence=min_confidence,
            since=since,
            page=page,
            page_size=page_size
        )
        
        # Convert to response models
        signal_responses = [
            SignalResponse(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                strength=signal.strength,
                source=signal.source,
                current_price=signal.current_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_score=signal.risk_score,
                reasoning=signal.reasoning,
                created_at=signal.created_at,
                expires_at=signal.expires_at
            )
            for signal in signals
        ]
        
        return SignalListResponse(
            signals=signal_responses,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(page * page_size) < total
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve signals: {str(e)}"
        )


@router.get("/latest/{symbol}", response_model=List[SignalResponse])
async def get_latest_signals(
    symbol: str,
    limit: int = Query(10, ge=1, le=50, description="Number of latest signals"),
    signal_service: SignalService = Depends(get_signal_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the latest signals for a symbol.
    
    - **symbol**: Stock symbol
    - **limit**: Maximum number of signals to return
    """
    try:
        signals = await signal_service.get_latest_signals(
            symbol=symbol.upper(),
            limit=limit
        )
        
        return [
            SignalResponse(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                strength=signal.strength,
                source=signal.source,
                current_price=signal.current_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_score=signal.risk_score,
                reasoning=signal.reasoning,
                created_at=signal.created_at,
                expires_at=signal.expires_at
            )
            for signal in signals
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest signals: {str(e)}"
        )


@router.get("/analytics/{symbol}", response_model=SignalAnalyticsResponse)
async def get_signal_analytics(
    symbol: str,
    days: int = Query(30, ge=1, le=365, description="Days to analyze"),
    signal_service: SignalService = Depends(get_signal_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get analytics and performance metrics for signals.
    
    - **symbol**: Stock symbol to analyze
    - **days**: Number of days to include in analysis
    """
    try:
        analytics = await signal_service.get_signal_analytics(
            symbol=symbol.upper(),
            days=days
        )
        
        return SignalAnalyticsResponse(**analytics)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve signal analytics: {str(e)}"
        )


@router.get("/stream/{symbol}")
async def get_signal_stream(
    symbol: str,
    limit: int = Query(50, ge=1, le=100, description="Number of signals in stream"),
    signal_service: SignalService = Depends(get_signal_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get real-time signal stream for a symbol.
    
    - **symbol**: Stock symbol
    - **limit**: Maximum number of signals in stream
    """
    try:
        signals = await signal_service.get_signal_stream(
            symbol=symbol.upper(),
            limit=limit
        )
        
        return {
            "symbol": symbol.upper(),
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve signal stream: {str(e)}"
        )


@router.post("/{symbol}/feedback")
async def submit_signal_feedback(
    symbol: str,
    signal_id: str,
    was_profitable: bool,
    actual_return: Optional[float] = None,
    notes: Optional[str] = None,
    signal_service: SignalService = Depends(get_signal_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Submit feedback on signal performance for machine learning improvement.
    
    - **symbol**: Stock symbol
    - **signal_id**: ID of the signal to provide feedback on
    - **was_profitable**: Whether the signal was profitable
    - **actual_return**: Actual return percentage (optional)
    - **notes**: Additional notes (optional)
    """
    try:
        await signal_service.update_signal_performance(
            signal_id=signal_id,
            was_profitable=was_profitable,
            actual_return=actual_return,
            notes=notes,
            user_id=current_user.get("user_id")
        )
        
        return {
            "status": "success",
            "message": "Signal feedback submitted successfully",
            "signal_id": signal_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit signal feedback: {str(e)}"
        )


@router.get("/latest", response_model=List[Signal])
async def get_latest_signals(
    limit: int = Query(default=10, ge=1, le=100),
    signal_service: SignalService = Depends(get_signal_service)
):
    """
    Get the latest trading signals
    
    This endpoint is used by the frontend dashboard to display recent signals.
    """
    # For MVP, generate mock signals if database is empty
    try:
        signals = await signal_service.get_latest_signals(limit)
        
        # If no signals in database, generate some mock ones
        if not signals or len(signals) == 0:
            mock_signals = []
            symbols = ['AAPL', 'GOOGL', 'TSLA', 'NVDA', 'MSFT', 'META', 'AMZN', 'SPY']
            signal_types = ['BUY', 'SELL', 'HOLD']
            strengths = ['STRONG', 'MODERATE', 'WEAK']
            
            for i in range(min(limit, len(symbols))):
                import random
                signal = Signal(
                    signal_id=f"mock_{i}",
                    symbol=symbols[i],
                    signal_type=random.choice(signal_types),
                    confidence=random.uniform(0.6, 0.95),
                    strength=random.choice(strengths),
                    current_price=random.uniform(50, 500),
                    reasoning="Technical indicators suggest momentum shift",
                    indicators={
                        "RSI": random.uniform(30, 70),
                        "MACD": random.uniform(-5, 5),
                        "BB_Position": random.uniform(0, 1)
                    },
                    timestamp=datetime.now().isoformat()
                )
                mock_signals.append(signal)
            
            return mock_signals
        
        return signals
        
    except Exception as e:
        # Return mock data on any error for MVP
        return []


@router.get("/{symbol}", response_model=Signal)
async def get_signal_for_symbol(
    symbol: str,
    signal_service: SignalService = Depends(get_signal_service)
):
    """Get the latest signal for a specific symbol"""
    signal = await signal_service.get_latest_signal_for_symbol(symbol.upper())
    
    if not signal:
        # Generate a mock signal for MVP
        import random
        return Signal(
            signal_id=f"mock_{symbol}",
            symbol=symbol.upper(),
            signal_type=random.choice(['BUY', 'SELL', 'HOLD']),
            confidence=random.uniform(0.6, 0.95),
            strength=random.choice(['STRONG', 'MODERATE', 'WEAK']),
            current_price=random.uniform(50, 500),
            reasoning="Technical analysis indicates opportunity",
            timestamp=datetime.now().isoformat()
        )
    
    return signal


@router.post("/", response_model=Signal)
async def create_signal(
    signal: SignalCreate,
    signal_service: SignalService = Depends(get_signal_service)
):
    """Create a new trading signal"""
    return await signal_service.create_signal(signal) 