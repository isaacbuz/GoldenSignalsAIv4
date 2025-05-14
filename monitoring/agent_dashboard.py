# monitoring/agent_dashboard.py
# Purpose: Implements a live agent dashboard for GoldenSignalsAI using Dash, displaying
# trading signals, agent activity, and market data for options trading.

import logging

import dash
import pandas as pd
import plotly.express as px
import requests
from dash import dcc, html
from dash.dependencies import Input, Output

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(
    [
        html.H1("GoldenSignalsAI Agent Dashboard"),
        dcc.Interval(
            id="interval-component",
            interval=60 * 1000,  # Update every 60 seconds
            n_intervals=0,
        ),
        html.Label("Select Symbol:"),
        dcc.Dropdown(
            id="symbol-dropdown",
            options=[
                {"label": "AAPL", "value": "AAPL"},
                {"label": "GOOGL", "value": "GOOGL"},
                {"label": "MSFT", "value": "MSFT"},
            ],
            value="AAPL",
        ),
        dcc.Graph(id="price-chart"),
        html.Div(id="prediction-output"),
    ]
)


@app.callback(
    [Output("price-chart", "figure"), Output("prediction-output", "children")],
    [Input("interval-component", "n_intervals"), Input("symbol-dropdown", "value")],
)
def update_dashboard(n_intervals, symbol):
    """Update the dashboard with the latest data.

    Args:
        n_intervals: Number of intervals elapsed (for auto-update)
        symbol: Selected stock symbol

    Returns:
        tuple: Updated chart figure and prediction text
    """
    logger.info({"message": f"Updating dashboard for {symbol}"})
    try:
        # Fetch dashboard data from API
        response = requests.get(f"http://localhost:8000/dashboard/{symbol}")
        response.raise_for_status()
        data = response.json()

        # Placeholder for price chart data
        df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2025-05-01", periods=30, freq="D"),
                "Price": [100 + i + np.random.randn() * 2 for i in range(30)],
            }
        )
        fig = px.line(df, x="Date", y="Price", title=f"{symbol} Price Trend")

        # Fetch prediction
        pred_response = requests.post(
            "http://localhost:8000/predict", json={"symbol": symbol}
        )
        pred_response.raise_for_status()
        prediction = pred_response.json()

        prediction_text = f"Prediction for {symbol}: {prediction['status']}"
        return fig, prediction_text
    except Exception as e:
        logger.error({"message": f"Failed to update dashboard: {str(e)}"})
        return px.line(title="Error"), f"Error: {str(e)}"


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
