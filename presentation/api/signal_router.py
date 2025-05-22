from fastapi import APIRouter, HTTPException
from infrastructure.data_fetcher import MarketDataFetcher
from domain.models.factory import ModelFactory
from domain.signal_engine import SignalEngine

router = APIRouter()

data_fetcher = MarketDataFetcher()
agent_factory = ModelFactory()  # Assuming ModelFactory is used for agent creation
signal_engine = SignalEngine(factory=agent_factory, data_fetcher=data_fetcher)

@router.get("/signal/{symbol}")
async def get_signal(symbol: str):
    try:
        result = signal_engine.generate_trade_signal(symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
