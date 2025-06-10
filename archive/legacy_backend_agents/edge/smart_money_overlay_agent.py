import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class SmartMoneyOverlayAgent(BaseAgent):
    def run(self, options_flow: pd.DataFrame) -> Dict[str, Any]:
        if options_flow.empty or not {'type', 'size', 'direction', 'iv_change'}.issubset(options_flow.columns):
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "explanation": "Insufficient smart money data"
            }

        bullish_flows = options_flow[
            (options_flow['type'] == 'call') & 
            (options_flow['direction'] == 'buy') & 
            (options_flow['size'] > 1000)
        ]
        bearish_flows = options_flow[
            (options_flow['type'] == 'put') & 
            (options_flow['direction'] == 'buy') & 
            (options_flow['size'] > 1000)
        ]

        bullish_score = bullish_flows['iv_change'].sum()
        bearish_score = bearish_flows['iv_change'].sum()

        if bullish_score > bearish_score:
            return {
                "signal": "bullish",
                "confidence": min(bullish_score / (abs(bearish_score) + 1e-5), 1.0),
                "explanation": f"Smart money leaning bullish: IV change net +{bullish_score:.2f}"
            }
        elif bearish_score > bullish_score:
            return {
                "signal": "bearish",
                "confidence": min(bearish_score / (abs(bullish_score) + 1e-5), 1.0),
                "explanation": f"Smart money leaning bearish: IV change net -{bearish_score:.2f}"
            }

        return {
            "signal": "neutral",
            "confidence": 0.3,
            "explanation": "Smart money flows balanced"
        }
