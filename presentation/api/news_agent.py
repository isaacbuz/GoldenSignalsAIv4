from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from infrastructure.auth.jwt_utils import verify_jwt_token
import os

from agents.sentiment.news_agent import NewsSentimentAgent

router = APIRouter(prefix="/api/v1/news", tags=["news"])

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "demo-key")

class NewsRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"

class AnalyzedHeadline(BaseModel):
    headline: str
    sentiment: str
    score: float

class NewsResponse(BaseModel):
    headlines: List[AnalyzedHeadline]

@router.post("/feed")
async def news_feed(request: NewsRequest, user=Depends(verify_jwt_token)):
    agent = NewsSentimentAgent(api_key=NEWS_API_KEY)
    try:
        results = agent.fetch_and_analyze(topic=request.symbol)
        return {"headlines": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
