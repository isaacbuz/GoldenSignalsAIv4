from fastapi import APIRouter, Request
from agents.grok.grok_backtest import GrokBacktestCritic
import os

router = APIRouter()

grok_api_key = os.getenv("GROK_API_KEY")
grok_critic = GrokBacktestCritic(grok_api_key)

@router.get("/grok/feedback")
async def get_grok_feedback(request: Request):
    # For demo, use hardcoded logic and metrics
    logic = "Buy if EMA9 > price and RSI < 30"
    win_rate = 62.5
    avg_return = 7.4
    # Call GrokBacktestCritic (sync call for now)
    suggestions = grok_critic.critique(logic, win_rate, avg_return)
    return {"suggestions": suggestions}
