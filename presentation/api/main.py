from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from GoldenSignalsAI.application.services.data_service import DataService
from GoldenSignalsAI.application.services.model_service import ModelService
from GoldenSignalsAI.application.services.strategy_service import StrategyService
from GoldenSignalsAI.infrastructure.auth.jwt_utils import verify_jwt_token
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

app = FastAPI()
data_service = DataService()
model_service = ModelService()
strategy_service = StrategyService()

@app.on_event("startup")
async def startup():
    redis_instance = redis.from_url("redis://localhost:6379")
    await FastAPILimiter.init(redis_instance)

class SymbolRequest(BaseModel):
    symbol: str

@app.post("/predict", dependencies=[Depends(verify_jwt_token), Depends(RateLimiter(times=10, seconds=60))])
async def predict(request: SymbolRequest):
    symbol = request.symbol
    historical_df, news_articles, _ = data_service.fetch_all_data(symbol)
    if historical_df is None:
        raise HTTPException(status_code=400, detail="Failed to fetch data")
    X, y, scaler = data_service.preprocess_data(historical_df)
    model_service.train_lstm(X, y, symbol)
    lstm_pred = model_service.predict_lstm(symbol, X[-1], scaler)
    xgboost_pred = model_service.train_xgboost(historical_df, symbol)
    lightgbm_pred = model_service.train_lightgbm(historical_df, symbol)
    sentiment_score = model_service.analyze_sentiment(news_articles)
    predicted_changes = [((lstm_pred - historical_df['close'].iloc[-1]) / historical_df['close'].iloc[-1]) if lstm_pred else 0, xgboost_pred or 0, lightgbm_pred or 0, sentiment_score]
    avg_pred_change = sum(predicted_changes) / len(predicted_changes)
    return {"symbol": symbol, "predicted_change": avg_pred_change}

@app.post("/backtest", dependencies=[Depends(verify_jwt_token), Depends(RateLimiter(times=10, seconds=60))])
async def backtest(request: SymbolRequest):
    symbol = request.symbol
    historical_df, _, _ = data_service.fetch_all_data(symbol)
    if historical_df is None:
        raise HTTPException(status_code=400, detail="Failed to fetch data")
    X, y, scaler = data_service.preprocess_data(historical_df)
    model_service.train_lstm(X, y, symbol)
    lstm_pred = model_service.predict_lstm(symbol, X[-1], scaler)
    xgboost_pred = model_service.train_xgboost(historical_df, symbol)
    lightgbm_pred = model_service.train_lightgbm(historical_df, symbol)
    sentiment_score = model_service.analyze_sentiment(news_articles)
    predicted_changes = [((lstm_pred - historical_df['close'].iloc[-1]) / historical_df['close'].iloc[-1]) if lstm_pred else 0, xgboost_pred or 0, lightgbm_pred or 0, sentiment_score]
    avg_pred_change = sum(predicted_changes) / len(predicted_changes)
    backtest_result = strategy_service.backtest(symbol, historical_df, [avg_pred_change] * len(historical_df))
    return {"symbol": symbol, "backtest_result": backtest_result}
