
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
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# Sentry initialization (set SENTRY_DSN in your environment)
sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=1.0)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Trusted frontend origins
trusted_origins = [
    "https://yourfrontend.com",
    "http://localhost:3000"
]

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=trusted_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"}))

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Sentry ASGI middleware
app.add_middleware(SentryAsgiMiddleware)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from .admin_endpoints import router as admin_router
from .admin_user_management import router as admin_user_router
from .signal_endpoints import router as signal_router
from presentation.api.gdpr_endpoints import router as gdpr_router
from presentation.api.grok_feedback import router as grok_feedback_router
from .ohlcv_endpoints import router as ohlcv_router
from .ws_endpoints import router as ws_router
from .arbitrage_endpoints import router as arbitrage_router
# --- Feature Routers ---
from .signal_explain import router as signal_explain_router
from .regime_detector import router as regime_detector_router
from .agent_performance import router as agent_performance_router
from .news_agent import router as news_agent_router
from .backtest_playground import router as backtest_playground_router
from .alert_manager import router as alert_manager_router
from .audit_trail import router as audit_trail_router
from .watchlist import router as watchlist_router
import redis.asyncio as redis

# === New agent routers ===
from presentation.api.meta_signal import router as meta_signal_router
from presentation.api.gpt_model_copilot import router as gpt_model_copilot_router
from presentation.api.forecasting import router as forecasting_router
from presentation.api.strategy_selector import router as strategy_selector_router

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

AUDIT_LOG_FILE = os.getenv("ADMIN_AUDIT_LOG", os.path.join("logs", "admin_audit.log"))
os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)  # Use os.path.join for portability
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

@app.post("/api/v1/refresh")
async def refresh_token(request: Request, refresh_token: str = Cookie(None), user=Depends(verify_jwt_token)):
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
@app.get("/api/v1/dashboard/{symbol}")
async def dashboard(symbol: str = Path(...), user=Depends(verify_jwt_token)):
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
# --- API Routers by Microservice Domain ---
# Admin/Users
app.include_router(gdpr_router, prefix="/gdpr")
app.include_router(grok_feedback_router, prefix="/api")
app.include_router(admin_router, prefix="/api/v1/admin")
app.include_router(admin_user_router, prefix="/api/v1/admin/users")
# Signals/Analytics
app.include_router(signal_router, prefix="/api/v1/signal")
app.include_router(signal_explain_router, prefix="/api/v1/signal_explain")
app.include_router(regime_detector_router, prefix="/api/v1/regime")
app.include_router(agent_performance_router, prefix="/api/v1/agent_performance")
app.include_router(news_agent_router, prefix="/api/v1/news")
app.include_router(ohlcv_router, prefix="/api/v1/ohlcv")
app.include_router(backtest_playground_router, prefix="/api/v1/backtest_playground")
app.include_router(alert_manager_router, prefix="/api/v1/alerts")
app.include_router(audit_trail_router, prefix="/api/v1/audit")
app.include_router(watchlist_router, prefix="/api/v1/watchlist")
# Realtime/Websocket
app.include_router(ws_router, prefix="/api/v1/ws")
# Arbitrage
app.include_router(arbitrage_router, prefix="/api/v1/arbitrage")

# === New agent routers ===
app.include_router(meta_signal_router)
app.include_router(gpt_model_copilot_router)
app.include_router(forecasting_router)
app.include_router(strategy_selector_router)

# === Advanced model agent routers ===
from presentation.api.finbert_sentiment import router as finbert_sentiment_router
from presentation.api.lstm_forecast import router as lstm_forecast_router
from presentation.api.ml_classifier import router as ml_classifier_router
from presentation.api.rsi_macd import router as rsi_macd_router
from presentation.api.correlation import router as correlation_router

router = APIRouter()
router.include_router(finbert_sentiment_router)
router.include_router(lstm_forecast_router)
router.include_router(ml_classifier_router)
router.include_router(rsi_macd_router)
router.include_router(correlation_router)

app.include_router(router)

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
@app.post("/api/v1/tickers/validate")
async def validate_ticker(request: TickerValidationRequest, user=Depends(verify_jwt_token)):
    symbol = request.symbol
    # Try to fetch data for the symbol
    historical_df, _, _ = data_service.fetch_all_data(symbol)
    if historical_df is not None and not historical_df.empty:
        return {"valid": True, "symbol": symbol}
    return {"valid": False, "symbol": symbol}

# NOTE: Auth relaxed for testing. Restore Depends(verify_jwt_token), Depends(RateLimiter(...)) for production.
@app.post("/api/v1/predict")
async def predict(request: SymbolRequest, user=Depends(verify_jwt_token)):
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
@app.post("/api/v1/backtest")
async def backtest(request: SymbolRequest, user=Depends(verify_jwt_token)):
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

# presentation/api/main.py
# Purpose: Implements the FastAPI backend for GoldenSignalsAI, providing endpoints for
# predictions, dashboard data, user preferences, and health checks. Includes JWT authentication
# for secure access, optimized for options trading workflows.

import logging
import os
from datetime import datetime, timedelta
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="GoldenSignalsAI API")

# Security setup
from dotenv import load_dotenv
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # Loaded from .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock user database
users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("password1"),
        "disabled": False,
    }
}


class Token:
    access_token: str
    token_type: str


class User:
    username: str
    disabled: bool = False


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)
    return None


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username)
    if user is None:
        raise credentials_exception
    return user


@app.post("/token", response_model=Dict)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info({"message": f"User {user.username} logged in"})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/health")
async def health_check():
    logger.info({"message": "Health check endpoint called"})
    return {"status": "healthy"}


@app.post("/predict")
async def predict(data: Dict, current_user: User = Depends(get_current_user)):
    logger.info({"message": f"Prediction requested by {current_user.username}"})
    # Mock prediction response
    return {"status": "Prediction successful", "symbol": data.get("symbol", "AAPL")}


@app.get("/dashboard/{symbol}")
async def get_dashboard_data(symbol: str, current_user: User = Depends(get_current_user)):
    logger.info({"message": f"Dashboard data requested for {symbol} by {current_user.username}"})
    # Mock dashboard data with options metrics
    return {
        "symbol": symbol,
        "price": 150.0,
        "trend": "up",
        "options_data": {
            "iv": 0.3,
            "delta": 0.5,
            "gamma": 0.1,
            "theta": -0.02
        }
    }
