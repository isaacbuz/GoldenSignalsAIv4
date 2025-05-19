# signal_endpoints.py
# FastAPI endpoints for trading signals and chart indicators
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from .auth_middleware import verify_token
from application.services.data_service import DataService
from application.services.model_service import ModelService
from application.services.strategy_service import StrategyService
from application.services.decision_logger import DecisionLogger
from domain.trading.strategies.indicators import TechnicalIndicators
from domain.trading.indicators import Indicators
import pandas as pd
import os

router = APIRouter(prefix="/api/signal", tags=["signal"])

data_service = DataService()
model_service = ModelService()
strategy_service = StrategyService()
decision_logger = DecisionLogger()

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "D"

@router.get("/markers")
async def get_trade_markers(symbol: str, timeframe: str = "D"):
    """Return entry, take profit, and exit markers for a given symbol and timeframe."""
    log = decision_logger.get_decision_log()
    markers = []
    for entry in log:
        if entry.get('symbol') == symbol and entry.get('timeframe', 'D') == timeframe:
            markers.append({
                "type": "entry",
                "price": entry.get("entry_price"),
                "timestamp": entry.get("timestamp")
            })
            if entry.get("take_profit"):
                markers.append({
                    "type": "take_profit",
                    "price": entry.get("take_profit"),
                    "timestamp": entry.get("timestamp")
                })
            if entry.get("stop_loss"):
                markers.append({
                    "type": "exit",
                    "price": entry.get("stop_loss"),
                    "timestamp": entry.get("timestamp")
                })
    return {"symbol": symbol, "timeframe": timeframe, "markers": markers}

class IndicatorRequest(BaseModel):
    symbol: str
    timeframe: str = "D"
    indicators: list[str] = []

# NOTE: Auth relaxed for testing. Restore Depends(verify_token) for production.
@router.post("/indicators")
async def get_technical_indicators(request: IndicatorRequest):
    """Return technical indicators for a given symbol and timeframe."""
    # Fetch historical data
    historical_df, _, _ = await data_service.fetch_all_data(request.symbol)
    if historical_df is None or len(historical_df) < 30:
        return {"symbol": request.symbol, "timeframe": request.timeframe, "indicators": {}}
    # Compute indicators
    ti = TechnicalIndicators(historical_df)
    indicators = {}
    try:
        indicators["MA_Confluence"] = float(ti.moving_average(20).iloc[-1]) if hasattr(ti, 'moving_average') else None
        indicators["RSI"] = float(ti.rsi(14).iloc[-1]) if hasattr(ti, 'rsi') else None
        macd_line, signal_line, histogram = ti.macd(12, 26, 9)
        indicators["MACD_Strength"] = float(macd_line.iloc[-1]) if macd_line is not None else None
        indicators["VWAP_Score"] = float(ti.vwap().iloc[-1]) if hasattr(ti, 'vwap') else None
        # Volume spike: compare last volume to mean of last 20
        indicators["Volume_Spike"] = float(historical_df['volume'].iloc[-1] / historical_df['volume'].rolling(20).mean().iloc[-1])
    except Exception as e:
        indicators["error"] = str(e)
    return {"symbol": request.symbol, "timeframe": request.timeframe, "indicators": indicators}
