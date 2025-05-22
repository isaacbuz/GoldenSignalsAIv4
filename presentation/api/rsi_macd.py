from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any
import pandas as pd
import logging
from agents.rsi_macd_agent import RSIMACDAgent

logger = logging.getLogger(__name__)
router = APIRouter()
rsi_macd_agent = RSIMACDAgent()

class RSIMACDRequest(BaseModel):
    ohlcv: List[Dict[str, Any]] = Field(..., min_items=27, description="OHLCV data with at least 'close' column.")

class RSIMACDResponse(BaseModel):
    signal: Any

@router.post("/rsi_macd/predict", response_model=RSIMACDResponse)
async def rsi_macd_predict(request: RSIMACDRequest):
    logger.info(f"RSI+MACD /predict called with OHLCV length {len(request.ohlcv)}.")
    df = pd.DataFrame(request.ohlcv)
    signal = rsi_macd_agent.compute_signal(df)
    if isinstance(signal, dict) and "error" in signal:
        logger.error(f"RSI+MACD error: {signal['error']}")
        return RSIMACDResponse(signal=signal)
    return RSIMACDResponse(signal=signal)
