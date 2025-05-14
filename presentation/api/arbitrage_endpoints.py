# FastAPI endpoints for arbitrage monitoring and execution
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from arbitrage.agents import CrossExchangeArbitrageAgent, ArbitrageOpportunity
from arbitrage.execution import ArbitrageExecutor
import asyncio

router = APIRouter()

import os
from infrastructure.agent_data_sources import AlphaVantageAgent, FinnhubAgent, PolygonAgent

alpha_agent = AlphaVantageAgent(os.getenv("ALPHA_VANTAGE_API_KEY"))
finnhub_agent = FinnhubAgent(os.getenv("FINNHUB_API_KEY"))
polygon_agent = PolygonAgent(os.getenv("POLYGON_API_KEY"))

def fetch_price_venue1(symbol):
    return alpha_agent.fetch_price_data(symbol)
def fetch_price_venue2(symbol):
    return finnhub_agent.fetch_price_data(symbol)
def fetch_price_venue3(symbol):
    return polygon_agent.fetch_price_data(symbol)

price_fetchers = {
    "AlphaVantage": fetch_price_venue1,
    "Finnhub": fetch_price_venue2,
    "Polygon": fetch_price_venue3,
}

arbitrage_agent = CrossExchangeArbitrageAgent(price_fetchers)
# Dummy executor (replace with real broker APIs)
broker_apis = {venue: object() for venue in price_fetchers.keys()}
arbitrage_executor = ArbitrageExecutor(broker_apis)

class ArbitrageRequest(BaseModel):
    symbol: str
    min_spread: float = 0.01

@router.post("/arbitrage/opportunities")
async def get_arbitrage_opportunities(request: ArbitrageRequest):
    opps = arbitrage_agent.find_opportunities(request.symbol, request.min_spread)
    return [opp.to_dict() for opp in opps]

class ExecuteRequest(BaseModel):
    symbol: str
    min_spread: float = 0.01

@router.post("/arbitrage/execute")
async def execute_arbitrage(request: ExecuteRequest):
    opps = arbitrage_agent.find_opportunities(request.symbol, request.min_spread)
    if not opps:
        raise HTTPException(status_code=404, detail="No arbitrage opportunities found")
    count = arbitrage_executor.execute_batch(opps)
    return {"executed": count, "total": len(opps)}

# Background task for continuous monitoring
arbitrage_opportunity_buffer = []

async def arbitrage_monitor(symbol: str, interval: int = 10):
    while True:
        opps = arbitrage_agent.find_opportunities(symbol)
        if opps:
            arbitrage_opportunity_buffer.clear()
            arbitrage_opportunity_buffer.extend([opp.to_dict() for opp in opps])
        await asyncio.sleep(interval)

@router.get("/arbitrage/live")
async def get_live_arbitrage():
    return arbitrage_opportunity_buffer
