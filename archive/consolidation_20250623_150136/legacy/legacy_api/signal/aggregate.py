# API endpoint: /api/signal/aggregate
# Runs all backend indicator agents and returns unified signal output for a given market input
from fastapi import APIRouter, Request
from src.domain.signal_engine import SignalEngine
from src.ml.models.factory import ModelFactory
from src.data.data_fetcher import MarketDataFetcher
import asyncio

router = APIRouter()

@router.post("/signal/aggregate")
async def aggregate_signal(request: Request):
    data = await request.json()
    symbol = data.get("symbol")
    factory = ModelFactory()
    data_fetcher = MarketDataFetcher()
    engine = SignalEngine(factory, data_fetcher)
    signal = await engine.generate_trade_signal(symbol)
    return {"status": "aggregated", "signal": signal.dict() if hasattr(signal, 'dict') else signal}
