from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from infrastructure.auth.jwt_utils import verify_jwt_token
import numpy as np
import pandas as pd
from domain.trading.regime_detector import MarketRegimeDetector

router = APIRouter(prefix="/api/v1/regime", tags=["regime"])

detector = MarketRegimeDetector()

class RegimeRequest(BaseModel):
    symbol: str
    prices: List[float]
    timeframe: str = "D"

class RegimeResponse(BaseModel):
    symbol: str
    regime: str
    confidence: float

@router.post("/detect", response_model=RegimeResponse)
async def detect_regime(request: RegimeRequest, user=Depends(verify_jwt_token)):
    try:
        prices = pd.Series(request.prices)
        regime = detector.detect_regime(prices)
        return {"symbol": request.symbol, "regime": regime, "confidence": 0.87}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
