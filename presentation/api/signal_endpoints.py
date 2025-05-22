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

from infrastructure.auth.jwt_utils import verify_jwt_token
router = APIRouter(prefix="/api/v1/signal", tags=["signal"])

data_service = DataService()
model_service = ModelService()
strategy_service = StrategyService()
decision_logger = DecisionLogger()

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "D"

@router.get("/markers")
async def get_trade_markers(symbol: str, timeframe: str = "D", user=Depends(verify_jwt_token)):
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
async def get_technical_indicators(request: IndicatorRequest, user=Depends(verify_jwt_token)):
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


# --- Unified Trade Suggestion Endpoint ---
from domain.models.signal import TradingSignal
from fastapi import Body
from datetime import datetime

class TradeSuggestionRequest(BaseModel):
    symbol: str
    timeframe: str = "D"
    risk_profile: str = "balanced"

@router.post("/trade_suggestion")
async def trade_suggestion(request: TradeSuggestionRequest = Body(...), user=Depends(verify_jwt_token)):
    """
    Returns a unified trade suggestion for a given symbol and timeframe.
    Includes direction (up/down), action (buy_call/buy_put/hold), confidence, entry_price, stop_loss, take_profit, and rationale.
    """
    # Fetch historical data
    historical_df, _, _ = await data_service.fetch_all_data(request.symbol)
    if historical_df is None or len(historical_df) < 30:
        raise HTTPException(status_code=404, detail="Not enough historical data for prediction.")

    # Prepare data for model
    X, y, scaler = await data_service.preprocess_data(historical_df)
    if X is None or y is None:
        raise HTTPException(status_code=500, detail="Data preprocessing failed.")

    # Get model prediction (ensemble average)
    pred_lstm = await model_service.predict_lstm(request.symbol, X[-1:], scaler)
    pred_xgb = await model_service.train_xgboost(historical_df, request.symbol)
    pred_lgb = await model_service.train_lightgbm(historical_df, request.symbol)
    predicted_change = float((pred_lstm + pred_xgb + pred_lgb) / 3)
    direction = "up" if predicted_change > 0 else "down" if predicted_change < 0 else "neutral"
    confidence = min(1.0, max(0.0, abs(predicted_change) / (abs(historical_df['close'].pct_change().std()) + 1e-6)))

    # Suggest call/put/hold
    if direction == "up":
        action = "buy_call"
    elif direction == "down":
        action = "buy_put"
    else:
        action = "hold"

    # Entry/exit/TP/SL
    latest_price = historical_df['close'].iloc[-1]
    entry_price = latest_price
    stop_loss = latest_price * (0.98 if direction == "up" else 1.02)
    take_profit = latest_price * (1.04 if direction == "up" else 0.96)

    # Rationale (simple for now)
    rationale = {
        "direction": direction,
        "predicted_change": predicted_change,
        "model_scores": {
            "lstm": float(pred_lstm),
            "xgboost": float(pred_xgb),
            "lightgbm": float(pred_lgb)
        }
    }

    # Build TradingSignal for validation
    signal = TradingSignal(
        symbol=request.symbol,
        action=action,
        confidence=confidence,
        ai_score=predicted_change,
        indicator_score=0.0,  # Optionally compute from technicals
        final_score=confidence,
        timestamp=datetime.utcnow().isoformat(),
        risk_profile=request.risk_profile,
        indicators=[],
        metadata=rationale
    )
    signal.validate_data()

    return {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "direction": direction,
        "action": action,
        "confidence": confidence,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rationale": rationale,
        "timestamp": signal.timestamp
    }
