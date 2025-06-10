# backend/orchestration/engine.py

from fastapi import APIRouter, Query
from archive.legacy_backend_agents.data_source_agent import DataSourceAgent
from archive.legacy_backend_agents.registry import AgentRegistry
from archive.legacy_backend_agents.blender_agent import BlenderAgent

router = APIRouter()

@router.get("/analyze")
def analyze_endpoint(ticker: str = Query(...), timeframe: str = Query("1D")):
    """
    HTTP endpoint for analyzing a ticker.
    """
    return analyze_ticker(ticker, timeframe)

def analyze_ticker(ticker: str, timeframe: str = "1D") -> dict:
    try:
        data = DataSourceAgent().run(ticker, timeframe)
        if not data or not data.get("price"):
            raise ValueError("Insufficient market data")

        agents = AgentRegistry()
        outputs = agents.run_all(data)
        if not outputs:
            raise ValueError("Agent outputs are empty")

        blended = BlenderAgent().blend(outputs)
        return {
            "ticker": ticker,
            **blended,
            "supporting_signals": outputs
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e),
            "supporting_signals": {},
            "confidence": 0,
            "strategy": None,
            "entry": None,
            "exit": None,
            "explanation": "Signal generation failed due to an internal error"
        }
