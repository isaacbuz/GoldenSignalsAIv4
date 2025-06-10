from fastapi import APIRouter, Request
from src.domain.signal_engine import SignalEngine
from src.domain.models.factory import ModelFactory
from infrastructure.data_fetcher import MarketDataFetcher
import asyncio

router = APIRouter()

@router.post("/signal/save")
async def save_signal(request: Request):
    data = await request.json()
    symbol = data.get("symbol")
    factory = ModelFactory()
    data_fetcher = MarketDataFetcher()
    engine = SignalEngine(factory, data_fetcher)
    signal = await engine.generate_trade_signal(symbol)
    # Save logic here (e.g., to DB)
    return {"status": "saved", "signal": signal.dict() if hasattr(signal, 'dict') else signal}
