from fastapi import APIRouter, Query
from backend.agents.ml.sentiment_aggregator import fetch_all_sentiment, recommend_top_sentiment_stocks

router = APIRouter()

@router.get("/sentiment")
async def get_sentiment(symbol: str = Query(..., example="AAPL")):
    return await fetch_all_sentiment(symbol)

@router.get("/sentiment/recommend")
def get_sentiment_recommendations(
    symbols: list[str] = Query(..., example=["AAPL", "TSLA", "MSFT"]),
    direction: str = Query("bullish")
):
    return recommend_top_sentiment_stocks(symbols, direction=direction)

@router.get("/sentiment/history")
async def get_sentiment_history(symbol: str = Query(..., example="AAPL")):
    """
    Fetch historical sentiment for a symbol. Returns list of sentiment scores over time.
    """
    try:
        from backend.db.models import SentimentScore
        from backend.db.session import get_db
        db = next(get_db())
        records = db.query(SentimentScore).filter(SentimentScore.symbol == symbol).order_by(SentimentScore.updated_at.desc()).limit(100).all()
        return [
            {
                "score": r.score,
                "trend": r.trend,
                "platform_breakdown": r.platform_breakdown,
                "sample_post": r.sample_post,
                "updated_at": r.updated_at.isoformat()
            }
            for r in records
        ]
    except Exception as e:
        return {"error": f"History unavailable: {e}"}
