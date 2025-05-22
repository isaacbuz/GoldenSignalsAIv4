from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Any
import pandas as pd
import logging
from agents.ml_classifier_agent import MLClassifierAgent

logger = logging.getLogger(__name__)
router = APIRouter()
ml_agent = MLClassifierAgent()

class MLClassifierRequest(BaseModel):
    features: List[List[float]] = Field(..., min_items=1, description="2D feature array for prediction.")

class MLClassifierResponse(BaseModel):
    signal: Any

@router.post("/ml_classifier/predict", response_model=MLClassifierResponse)
async def ml_classifier_predict(request: MLClassifierRequest):
    logger.info(f"MLClassifier /predict called with features shape ({len(request.features)}, {len(request.features[0]) if request.features else 0}).")
    features = pd.DataFrame(request.features)
    signal = ml_agent.predict_signal(features)
    if isinstance(signal, dict) and "error" in signal:
        logger.error(f"MLClassifier error: {signal['error']}")
        return MLClassifierResponse(signal=signal)
    return MLClassifierResponse(signal=signal)
