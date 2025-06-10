from fastapi import APIRouter, WebSocket
from src.domain.signal_engine import SignalEngine
from src.domain.models.factory import ModelFactory
from infrastructure.data_fetcher import MarketDataFetcher
import asyncio

router = APIRouter()

@router.websocket("/ws/signals/{symbol}")
async def signals_ws(websocket: WebSocket, symbol: str):
    await websocket.accept()
    factory = ModelFactory()
    data_fetcher = MarketDataFetcher()
    engine = SignalEngine(factory, data_fetcher)
    while True:
        signal = await engine.generate_trade_signal(symbol)
        await websocket.send_json(signal.dict() if hasattr(signal, 'dict') else signal)
        await asyncio.sleep(60)
