from fastapi import APIRouter, Request
from typing import List
import asyncio
from archive.legacy_backend_agents.arbitrage.parallel_arb_scanner import scan_all
from archive.legacy_backend_agents.arbitrage.etf_nav_agent import ETFNavAgent
from archive.legacy_backend_agents.arbitrage.stat_arb_agent import StatArbAgent
from archive.legacy_backend_agents.arbitrage.arbitrage_executor import ArbitrageExecutor
from archive.legacy_backend_agents.arbitrage.sim_log_store import SimLogStore

router = APIRouter()

# Example symbols/venues; in production, make this dynamic/configurable
SYMBOLS = ["AAPL", "TSLA", "NVDA", "SPY"]
VENUES = ["NYSE", "BATS", "ARCA"]

@router.get("/api/arbitrage")
def get_arbitrage_opportunities():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(scan_all(SYMBOLS, VENUES))
    loop.close()
    return {"opportunities": results}

@router.post("/api/arbitrage")
async def arbitrage_handler(request: Request):
    payload = await request.json()
    symbols = payload.get("symbols", SYMBOLS)
    venues = payload.get("venues", VENUES)
    nav_data = payload.get("etf_nav_data", {})
    pair_data = payload.get("cointegrated_assets", {})

    parallel_results = await scan_all(symbols, venues)

    # ETF NAV Agent
    etf_result = []
    for symbol, values in nav_data.items():
        result = ETFNavAgent().run(values)
        result["symbol"] = symbol
        etf_result.append(result)

    # Stat Arb Agent
    stat_result = []
    for pair_name, pair_series in pair_data.items():
        result = StatArbAgent().run(pair_series)
        result["pair"] = pair_name
        stat_result.append(result)

    return {
        "opportunities": parallel_results,
        "etf_signals": etf_result,
        "stat_arb_signals": stat_result
    }

@router.post("/api/arbitrage/execute")
async def arbitrage_execute(request: Request):
    payload = await request.json()
    opportunity = payload.get("opportunity")
    orderbook = payload.get("orderbook")
    executor = ArbitrageExecutor()
    result = executor.simulate(opportunity, orderbook)
    SimLogStore.append({**result, "opportunity": opportunity})
    return result

@router.get("/api/arbitrage/history")
def arbitrage_history():
    return {"history": SimLogStore.get_all()}
