import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
from domain.trading.indicators import Indicators
from domain.trading.strategies.signal_engine import SignalEngine
from application.monitoring.health_monitor import AIMonitor
from application.services.decision_logger import DecisionLogger

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
indicators = Indicators(data)
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

    monitor.update_metrics()
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
    app.run(debug=True, port=8050)
