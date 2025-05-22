from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any
import logging
from agents.finbert_sentiment_agent import FinBERTSentimentAgent

logger = logging.getLogger(__name__)
router = APIRouter()
finbert_agent = FinBERTSentimentAgent()

class FinBERTRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze.")

class FinBERTResponse(BaseModel):
    average_score: float
    raw_results: Any

@router.post("/finbert_sentiment/analyze", response_model=FinBERTResponse)
async def analyze_sentiment(request: FinBERTRequest):
    logger.info(f"FinBERT /analyze called with {len(request.texts)} texts.")
    result = finbert_agent.analyze_texts(request.texts)
    if "error" in result:
        logger.error(f"FinBERT error: {result['error']}")
        return FinBERTResponse(average_score=0.0, raw_results=[{"error": result["error"]}])
    return FinBERTResponse(**result)
