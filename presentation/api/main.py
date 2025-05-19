from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from application.services.data_service import DataService
from application.services.model_service import ModelService
from application.services.strategy_service import StrategyService
from infrastructure.auth.jwt_utils import verify_jwt_token
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .admin_endpoints import router as admin_router
from .admin_user_management import router as admin_user_router
from .signal_endpoints import router as signal_router
from .ohlcv_endpoints import router as ohlcv_router
from .ws_endpoints import router as ws_router
from .arbitrage_endpoints import router as arbitrage_router
import redis.asyncio as redis
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from .rate_limit import limiter
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
load_dotenv()
import os

AUDIT_LOG_FILE = os.getenv("ADMIN_AUDIT_LOG", "./logs/admin_audit.log")
os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
if not logging.getLogger("audit").handlers:
    audit_handler = RotatingFileHandler(AUDIT_LOG_FILE, maxBytes=2*1024*1024, backupCount=5)
    audit_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(audit_handler)

import logging
import traceback
from fastapi.responses import JSONResponse
from fastapi.requests import Request

app = FastAPI()

# Allow CORS from frontend (Vite) and backend (API) during development
origins = [
    "http://localhost:3000",  # Vite dev server
    "http://localhost:8001",  # FastAPI backend
    "http://localhost:8080",  # API microservice (if used)
    "https://your-production-domain.com"
]
logging.info(f"Setting up CORS with origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.middleware("http")
async def log_cors_requests(request: Request, call_next):
    if request.method == "OPTIONS" or request.headers.get("origin"):
        logging.info(f"CORS/Preflight request: {request.method} {request.url} Origin={request.headers.get('origin')}")
    response = await call_next(request)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logging.error(f"[GLOBAL EXCEPTION] {exc}\n{tb}")
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return {"detail": "Rate limit exceeded"}

from fastapi import Request, Response, Cookie, Path
from infrastructure.auth.jwt_utils import create_access_token, decode_refresh_token

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/refresh")
async def refresh_token(request: Request, refresh_token: str = Cookie(None)):
    """
    Refresh the JWT access token using a valid refresh token (from cookie).
    Returns a new access token if the refresh token is valid and not expired.
    """
    if not refresh_token:
        return Response(status_code=401, content="Missing refresh token.")
    try:
        payload = decode_refresh_token(refresh_token)
        user_id = payload.get("sub")
        if not user_id:
            return Response(status_code=401, content="Invalid refresh token.")
        # Issue new access token
        access_token = create_access_token({"sub": user_id})
        return {"access_token": access_token}
    except Exception as e:
        return Response(status_code=401, content=f"Invalid or expired refresh token: {str(e)}")


# --- Dashboard endpoint for frontend ---
@app.get("/dashboard/{symbol}")
async def dashboard(symbol: str = Path(...)):
    import traceback
    try:
        historical_df, news_articles, _ = await data_service.fetch_all_data(symbol)
        if historical_df is None or historical_df.empty:
            return {"symbol": symbol, "error": "No data found"}
        # Example: get last close price and simple stats
        last_close = float(historical_df['close'].iloc[-1]) if 'close' in historical_df.columns else None
        avg_volume = float(historical_df['volume'].mean()) if 'volume' in historical_df.columns else None
        # Add more advanced analytics as needed
        return {
            "symbol": symbol,
            "last_close": last_close,
            "avg_volume": avg_volume,
            "message": "This is a test dashboard endpoint. Replace with real data aggregation."
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"DASHBOARD ERROR: {e}\n{tb}")
        return {"symbol": symbol, "error": str(e), "traceback": tb}

# Mount admin endpoints
app.include_router(admin_router, prefix="/api/admin")
app.include_router(admin_user_router)
app.include_router(signal_router)
app.include_router(ohlcv_router)
app.include_router(ws_router)
app.include_router(arbitrage_router, prefix="/api")

data_service = DataService()
model_service = ModelService()
strategy_service = StrategyService()

from infrastructure.env_validator import env_validator
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

class TickerValidationRequest(BaseModel):
    symbol: str

# NOTE: Auth relaxed for testing. Restore Depends(verify_jwt_token), Depends(RateLimiter(...)) for production.
@app.post("/api/tickers/validate")
async def validate_ticker(request: TickerValidationRequest):
    symbol = request.symbol
    # Try to fetch data for the symbol
    historical_df, _, _ = data_service.fetch_all_data(symbol)
    if historical_df is not None and not historical_df.empty:
        return {"valid": True, "symbol": symbol}
    return {"valid": False, "symbol": symbol}

# NOTE: Auth relaxed for testing. Restore Depends(verify_jwt_token), Depends(RateLimiter(...)) for production.
@app.post("/predict")
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

# NOTE: Auth relaxed for testing. Restore Depends(verify_jwt_token), Depends(RateLimiter(...)) for production.
@app.post("/backtest")
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
