#!/usr/bin/env python3
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = "/Users/isaacbuz/Documents/Projects/GoldenSignalsAI"

dirs = [
    "presentation/api",
    "presentation/frontend/app",
    "presentation/frontend/src",
    "presentation/tests/orchestration",
    "presentation/tests/__pycache__"
]

files = {
    "presentation/api/Dockerfile": """FROM python:3.10-slim
WORKDIR /app
COPY presentation/api/requirements.txt .
RUN pip install -r requirements.txt
COPY presentation/api/ .
COPY infrastructure/auth/ infrastructure/auth/
COPY infrastructure/config/ infrastructure/config/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "presentation/api/main.py": """from fastapi import FastAPI, Depends, HTTPException
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
""",
    "presentation/api/requirements.txt": """fastapi==0.115.0
uvicorn==0.30.6
python-jose==3.3.0
python-dotenv==1.0.1
fastapi-limiter==0.1.6
""",
    "presentation/api/auth_middleware.py": """from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from GoldenSignalsAI.infrastructure.auth.jwt_utils import verify_jwt_token

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not verify_jwt_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token
""",
    "presentation/frontend/app/__init__.py": "# presentation/frontend/app/__init__.py\n",
    "presentation/frontend/app/dashboard.py": """import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
from GoldenSignalsAI.domain.trading.indicators import TechnicalIndicators
from GoldenSignalsAI.domain.trading.strategies.signal_engine import SignalEngine
from GoldenSignalsAI.application.monitoring.health_monitor import AIMonitor
from GoldenSignalsAI.application.services.decision_logger import DecisionLogger

app = dash.Dash(__name__)
dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=1440, freq="T")
data = pd.DataFrame({
    "time": dates,
    "close": [280 + i * 0.01 for i in range(1440)],
    "high": [281 + i * 0.01 for i in range(1440)],
    "low": [279 + i * 0.01 for i in range(1440)],
    "open": [280 + i * 0.01 for i in range(1440)],
    "volume": [1000000] * 1440
})
indicators = TechnicalIndicators(data)
monitor = AIMonitor()
logger = DecisionLogger()

app.layout = html.Div([
    html.H1("GoldenSignalsAI Dashboard"),
    html.H2("AI-Driven Stock Trading System"),
    dcc.Input(id="symbol-input", value="TSLA", type="text"),
    dcc.Dropdown(id="time-range", options=["1D", "1W", "1M", "3M"], value="1D"),
    dcc.Dropdown(id="risk-profile", options=["conservative", "balanced", "aggressive"], value="balanced"),
    html.Button("Generate Signal", id="signal-button"),
    dcc.Graph(id="price-chart"),
    dcc.Graph(id="rsi-chart"),
    dcc.Graph(id="macd-chart"),
    html.Div(id="signal-output"),
    html.Div(id="decision-explanation"),
    html.H3("System Health"),
    dcc.Graph(id="health-metrics"),
    html.H3("Decision Ledger"),
    dcc.Graph(id="decision-ledger")
])

@app.callback(
    [Output("price-chart", "figure"), Output("rsi-chart", "figure"), Output("macd-chart", "figure"),
     Output("signal-output", "children"), Output("decision-explanation", "children"),
     Output("health-metrics", "figure"), Output("decision-ledger", "figure")],
    [Input("symbol-input", "value"), Input("time-range", "value"), Input("risk-profile", "value"),
     Input("signal-button", "n_clicks")]
)
def update_dashboard(symbol, time_range, risk_profile, n_clicks):
    ma10 = indicators.moving_average(10)
    ma50 = indicators.moving_average(50)
    ma200 = indicators.moving_average(200)
    ema9 = indicators.exponential_moving_average(9)
    ema12 = indicators.exponential_moving_average(12)
    vwap = indicators.vwap()
    upper_band, middle_band, lower_band = indicators.bollinger_bands(20)
    rsi = indicators.rsi(14)
    macd_line, signal_line, histogram = indicators.macd(12, 26, 9)

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(x=data['time'], open=data['open'], high=data['high'], low=data['low'],
                                       close=data['close'], name="Price"))
    price_fig.add_trace(go.Scatter(x=data['time'], y=ma10, name="MA(10)", line=dict(color='blue')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=ma50, name="MA(50)", line=dict(color='pink')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=ma200, name="MA(200)", line=dict(color='lightblue')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=ema9, name="EMA(9)", line=dict(color='orange')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=ema12, name="EMA(12)", line=dict(color='purple')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=vwap, name="VWAP", line=dict(color='purple', dash='dash')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=upper_band, name="Bollinger Upper", line=dict(color='gray')))
    price_fig.add_trace(go.Scatter(x=data['time'], y=lower_band, name="Bollinger Lower", line=dict(color='gray')))
    price_fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Time", yaxis_title="Price")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=data['time'], y=rsi, name="RSI(14)", line=dict(color='orange')))
    rsi_fig.add_shape(type="line", x0=data['time'].iloc[0], x1=data['time'].iloc[-1], y0=70, y1=70,
                      line=dict(color="red", dash="dash"))
    rsi_fig.add_shape(type="line", x0=data['time'].iloc[0], x1=data['time'].iloc[-1], y0=30, y1=30,
                      line=dict(color="green", dash="dash"))
    rsi_fig.update_layout(title="RSI (14)", xaxis_title="Time", yaxis_title="RSI")

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=data['time'], y=macd_line, name="MACD", line=dict(color='blue')))
    macd_fig.add_trace(go.Scatter(x=data['time'], y=signal_line, name="Signal", line=dict(color='orange')))
    macd_fig.add_trace(go.Bar(x=data['time'], y=histogram, name="Histogram"))
    macd_fig.update_layout(title="MACD (12, 26, 9)", xaxis_title="Time", yaxis_title="MACD")

    signal_engine = SignalEngine(data)
    signal = signal_engine.generate_signal(symbol, risk_profile)
    signal_text = f"AI Signal: {signal['action']} at ${signal['price']:.2f}, Confidence Score: {signal['confidence_score']:.2f}"
    if signal["action"] == "Buy":
        signal_text += f", Stop Loss: ${signal['stop_loss']:.2f}, Profit Target: ${signal['profit_target']:.2f}"

    score = signal_engine.compute_signal_score()
    explanation = [
        html.H3("Why Did AI Trade?"),
        html.P(f"Composite Score: {score:.2f}"),
        html.Ul([
            html.Li(f"MA Cross (Weight: {signal_engine.weights['ma_cross']:.2f}): {'Bullish' if data['close'].iloc[-1] > ma10.iloc[-1] > ma50.iloc[-1] else 'Bearish'}"),
            html.Li(f"EMA Cross (Weight: {signal_engine.weights['ema_cross']:.2f}): {'Bullish' if ema9.iloc[-1] > ema12.iloc[-1] else 'Bearish'}"),
            html.Li(f"VWAP (Weight: {signal_engine.weights['vwap']:.2f}): {'Above' if data['close'].iloc[-1] > vwap.iloc[-1] else 'Below'}"),
            html.Li(f"Bollinger Bands (Weight: {signal_engine.weights['bollinger']:.2f}): {'Oversold' if data['close'].iloc[-1] < lower_band.iloc[-1] else 'Overbought' if data['close'].iloc[-1] > upper_band.iloc[-1] else 'Neutral'}"),
            html.Li(f"RSI (Weight: {signal_engine.weights['rsi']:.2f}): {rsi.iloc[-1]:.2f} ({'Oversold' if rsi.iloc[-1] < 30 else 'Overbought' if rsi.iloc[-1] > 70 else 'Neutral'})"),
            html.Li(f"MACD (Weight: {signal_engine.weights['macd']:.2f}): {'Bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'Bearish'}")
        ])
    ]

    await monitor.update_metrics()
    health_data = pd.DataFrame([
        {"Metric": metric, "Value": monitor.metrics_history[metric][-1]["value"] if monitor.metrics_history[metric] else 0}
        for metric in monitor.METRICS
    ])
    health_fig = px.bar(health_data, x="Metric", y="Value", title="System Health Metrics")

    decision_log = logger.get_decision_log()
    decision_log_df = pd.DataFrame(decision_log)
    decision_log_fig = px.scatter(decision_log_df, x="timestamp", y="confidence", color="action",
                                  title="Decision Ledger", hover_data=["symbol", "entry_price"]) if not decision_log_df.empty else px.scatter(title="Decision Ledger (No Decisions Yet)")

    return price_fig, rsi_fig, macd_fig, signal_text, explanation, health_fig, decision_log_fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
""",
    "presentation/frontend/app/login.py": """import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from GoldenSignalsAI.infrastructure.auth.twilio_mfa import TwilioMFA

if not firebase_admin._apps:
    cred = credentials.Certificate("path/to/firebase-credentials.json")
    firebase_admin.initialize_app(cred)

st.title("GoldenSignalsAI Login")
email = st.text_input("Email")
password = st.text_input("Password", type="password")
phone_number = st.text_input("Phone Number for MFA")

if st.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.session_state["user"] = user.uid
        mfa = TwilioMFA()
        verification_sid = mfa.send_mfa_code(phone_number)
        st.session_state["verification_sid"] = verification_sid
        st.session_state["phone_number"] = phone_number
        st.success("MFA code sent to your phone!")
    except:
        st.error("Invalid email or password")

if "user" in st.session_state:
    mfa_code = st.text_input("Enter MFA Code")
    if st.button("Verify MFA"):
        mfa = TwilioMFA()
        if mfa.verify_mfa_code(st.session_state["phone_number"], mfa_code):
            st.success("Logged in successfully!")
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid MFA code")

if "logged_in" in st.session_state and st.session_state["logged_in"]:
    st.write("Redirecting to dashboard...")
    st.experimental_rerun()
""",
    "presentation/frontend/src/__init__.py": "# presentation/frontend/src/__init__.py\n",
    "presentation/frontend/src/styles.css": """/* presentation/frontend/src/styles.css */
body { font-family: Arial, sans-serif; }
h1 { color: #1e3a8a; }
h2 { color: #4b5e7d; }
button { background-color: #2563eb; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
button:hover { background-color: #1d4ed8; }
""",
    "presentation/frontend/src/dark_mode.css": """/* presentation/frontend/src/dark_mode.css */
body.dark-mode { background-color: #1a202c; color: #e2e8f0; }
.dark-mode h1 { color: #60a5fa; }
.dark-mode h2 { color: #93c5fd; }
.dark-mode button { background-color: #3b82f6; }
.dark-mode button:hover { background-color: #2563eb; }
""",
    "presentation/frontend/src/utils.js": """// presentation/frontend/src/utils.js
function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
}
""",
    "presentation/frontend/Dockerfile": """FROM python:3.10-slim
WORKDIR /app
COPY presentation/frontend/ .
RUN pip install dash plotly pandas
CMD ["python", "app/dashboard.py"]
""",
    "presentation/tests/__init__.py": "# presentation/tests/__init__.py\n",
    "presentation/tests/conftest.py": """import pytest
import pandas as pd
from unittest.mock import patch
from GoldenSignalsAI.domain.trading.strategies.trading_env import TradingEnv
from GoldenSignalsAI.infrastructure.data.fetchers.database_fetcher import fetch_stock_data

@pytest.fixture
def symbol():
    return "AAPL"

@pytest.fixture(autouse=True)
def mock_fetch_stock_data():
    with patch('GoldenSignalsAI.infrastructure.data.fetchers.database_fetcher.fetch_stock_data') as mocked_fetch:
        mock_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [100 + i for i in range(100)],
            'volume': [1000000] * 100
        })
        mocked_fetch.return_value = mock_data
        yield mocked_fetch

@pytest.fixture
def stock_data(symbol, mock_fetch_stock_data):
    df = fetch_stock_data(symbol)
    if df is None:
        pytest.fail(f"Failed to fetch stock data for {symbol}")
    return df

@pytest.fixture
def trading_env(stock_data, symbol):
    try:
        return TradingEnv(stock_data, symbol)
    except Exception as e:
        pytest.fail(f"Failed to initialize TradingEnv: {str(e)}")
""",
    "presentation/tests/test.py": "def test_placeholder():\n    assert True\n",
    "presentation/tests/test_agentic_ai.py": "def test_agentic_ai_placeholder():\n    assert True\n",
    "presentation/tests/test_data_fetcher_edge_cases.py": "def test_data_fetcher_edge_cases_placeholder():\n    assert True\n",
    "presentation/tests/test_db.py": "def test_db_placeholder():\n    assert True\n",
    "presentation/tests/test_dependencies.py": "def test_dependencies_placeholder():\n    assert True\n",
    "presentation/tests/test_rl.py": "def test_rl_placeholder():\n    assert True\n",
    "presentation/tests/test_technical_indicators.py": "def test_technical_indicators_placeholder():\n    assert True\n",
    "presentation/tests/test_strategy_engine.py": "def test_strategy_engine_placeholder():\n    assert True\n",
    "presentation/tests/__pycache__/__init__.cpython-310.pyc": "# Compiled file, generated during runtime\n",
    "presentation/tests/__pycache__/conftest.cpython-310-pytest-7.4.4.pyc": "# Compiled file, generated during runtime\n",
    "presentation/tests/orchestration/orchestrator.py": """# This file is misplaced; should be in application/ai_service/orchestrator.py
def test_orchestrator_placeholder():
    assert True
"""
}

for directory in dirs:
    try:
        full_path = os.path.join(BASE_DIR, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")

for file_path, content in files.items():
    try:
        full_path = os.path.join(BASE_DIR, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        logger.info(f"Created file: {file_path}")
    except Exception as e:
        logger.error(f"Error creating file {file_path}: {str(e)}")

logger.info("Presentation files generation complete.")