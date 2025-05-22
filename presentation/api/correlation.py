from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from agents.correlation_agent import CorrelationAgent

router = APIRouter()
correlation_agent = CorrelationAgent()
logger = logging.getLogger(__name__)

class CorrelationRequest(BaseModel):
    data: Dict[str, List[float]] = Field(..., description="Dictionary of asset names to time series values.")
    method: Optional[str] = Field('pearson', description="Correlation method: 'pearson', 'spearman', or 'kendall'.")

    @root_validator
    def validate_data(cls, values):
        data = values.get('data')
        if not data or len(data) < 2:
            raise ValueError("At least two assets must be provided.")
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All series must be of equal length.")
        return values

class RollingCorrelationRequest(BaseModel):
    series1: List[float]
    series2: List[float]
    window: int = Field(30, description="Rolling window size.")
    method: Optional[str] = Field('pearson', description="Correlation method: 'pearson', 'spearman', or 'kendall'.")

    @root_validator
    def validate_series(cls, values):
        s1, s2, window = values.get('series1'), values.get('series2'), values.get('window')
        if not (s1 and s2) or len(s1) != len(s2):
            raise ValueError("Both series must be provided and of equal length.")
        if len(s1) < window:
            raise ValueError("Series length must be at least as large as window size.")
        return values

@router.post('/correlation/calculate')
def calculate_correlation(request: CorrelationRequest):
    try:
        df = pd.DataFrame(request.data)
        result = correlation_agent.compute_correlation(df, method=request.method)
        if 'error' in result:
            logger.error(f"Correlation error: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
        return result
    except Exception as e:
        logger.exception("API correlation calculation failed.")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/correlation/rolling')
def rolling_correlation(request: RollingCorrelationRequest):
    try:
        s1 = pd.Series(request.series1)
        s2 = pd.Series(request.series2)
        result = correlation_agent.compute_rolling_correlation(s1, s2, window=request.window, method=request.method)
        if 'error' in result:
            logger.error(f"Rolling correlation error: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
        return result
    except Exception as e:
        logger.exception("API rolling correlation calculation failed.")
        raise HTTPException(status_code=500, detail=str(e))
