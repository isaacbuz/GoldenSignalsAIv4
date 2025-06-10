from typing import Dict, Any
from ..base_agent import BaseAgent

class ETFNavAgent(BaseAgent):
    def run(self, data: Dict[str, float]) -> Dict[str, Any]:
        nav = data.get("nav")
        etf_price = data.get("etf_price")

        if nav is None or etf_price is None:
            return {
                "signal": "neutral",
                "confidence": 0,
                "explanation": "Missing NAV or ETF price"
            }

        delta = etf_price - nav
        spread = abs(delta)

        if spread > 0.5:
            return {
                "signal": "arbitrage",
                "confidence": min(spread / nav, 1.0),
                "explanation": f"ETF price deviates from NAV by ${delta:.2f}"
            }

        return {
            "signal": "neutral",
            "confidence": 0.3,
            "explanation": "ETF within NAV range"
        }
