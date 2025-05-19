# ohlcv_endpoints.py
from fastapi import APIRouter, HTTPException
from application.services.data_service import DataService
from pydantic import BaseModel
from typing import List

router = APIRouter()
data_service = DataService()

class OHLCVBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

import logging
import traceback
from fastapi.responses import JSONResponse

@router.get("/api/ohlcv")
async def get_ohlcv(symbol: str, timeframe: str = "D"):
    """Return OHLCV bars for a symbol and timeframe."""
    try:
        historical_df, _, _ = await data_service.fetch_all_data(symbol)
        if historical_df is None or len(historical_df) == 0:
            logging.error(f"No data found for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        # Optionally resample here based on timeframe
        bars = [
            {
                "timestamp": str(row[0]),
                "open": float(row[1]["open"]),
                "high": float(row[1]["high"]),
                "low": float(row[1]["low"]),
                "close": float(row[1]["close"]),
                "volume": float(row[1]["volume"])
            }
            for row in historical_df.reset_index().iterrows()
        ]
        return {"data": bars}
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"[get_ohlcv] Error for symbol={symbol}, timeframe={timeframe}: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})

@router.options("/api/ohlcv")
def ohlcv_cors_options():
    """CORS preflight diagnostic endpoint."""
    return JSONResponse(status_code=200, content={"message": "CORS preflight OK"})

