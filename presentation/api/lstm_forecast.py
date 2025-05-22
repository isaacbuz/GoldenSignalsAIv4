from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Any
import pandas as pd
import logging
from agents.lstm_forecast_agent import LSTMForecastAgent

logger = logging.getLogger(__name__)
router = APIRouter()
lstm_agent = LSTMForecastAgent()

class LSTMForecastRequest(BaseModel):
    series: List[float] = Field(..., min_items=21, description="Time series data for forecasting.")

class LSTMForecastResponse(BaseModel):
    prediction: Any

@router.post("/lstm_forecast/predict", response_model=LSTMForecastResponse)
async def lstm_forecast_predict(request: LSTMForecastRequest):
    logger.info(f"LSTM /predict called with series length {len(request.series)}.")
    series = pd.Series(request.series)
    prediction = lstm_agent.predict(series)
    if isinstance(prediction, dict) and "error" in prediction:
        logger.error(f"LSTM error: {prediction['error']}")
        return LSTMForecastResponse(prediction=prediction)
    return LSTMForecastResponse(prediction=prediction)
