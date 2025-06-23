from fastapi import APIRouter, Query
from agents.forecast_ensemble_agent import ForecastEnsembleAgent

router = APIRouter()
ensemble_agent = ForecastEnsembleAgent()

@router.get("/ensemble/forecast")
def get_ensemble_forecast(symbol: str = Query(..., example="AAPL")):
    from archive.legacy_backend_agents.ml.market_data_provider import MarketDataProvider
    provider = MarketDataProvider(symbol)
    try:
        # Fetch 30 days of daily close prices
        history = provider.get_history(period="1mo", interval="1d")
        prices = [bar['close'] for bar in history if 'close' in bar]
        if not prices or len(prices) < 10:
            return {"error": "Not enough historical data for symbol.", "symbol": symbol}
        market_data = {'close': prices}
        result = ensemble_agent.run(market_data)
        result['symbol'] = symbol
        return result
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
