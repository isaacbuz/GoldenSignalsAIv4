from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from GoldenSignalsAI.application.services.data_service import DataService
from GoldenSignalsAI.application.services.model_service import ModelService
from GoldenSignalsAI.application.services.strategy_service import StrategyService
from GoldenSignalsAI.infrastructure.auth.jwt_utils import verify_jwt_token
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .admin_endpoints import router as admin_router
from .admin_user_management import router as admin_user_router
import redis.asyncio as redis
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import logging
from logging.handlers import RotatingFileHandler
import os

AUDIT_LOG_FILE = os.getenv("ADMIN_AUDIT_LOG", "./logs/admin_audit.log")
os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
if not logging.getLogger("audit").handlers:
    audit_handler = RotatingFileHandler(AUDIT_LOG_FILE, maxBytes=2*1024*1024, backupCount=5)
    audit_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(audit_handler)

app = FastAPI()

origins = [
    "http://localhost:8080",
    "https://your-production-domain.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return {"detail": "Rate limit exceeded"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Mount admin endpoints
app.include_router(admin_router, prefix="/api/admin")
app.include_router(admin_user_router)

data_service = DataService()
model_service = ModelService()
strategy_service = StrategyService()

from GoldenSignalsAI.infrastructure.env_validator import env_validator
import sys

@app.on_event("startup")
async def startup():
    env_validator.log_api_key_statuses()
    if not env_validator.run_preflight_checks():
        import logging
        logging.critical("Startup aborted due to failed environment validation.")
        sys.exit(1)
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
