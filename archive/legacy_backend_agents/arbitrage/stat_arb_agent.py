import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.stattools import coint
from ..base_agent import BaseAgent

class StatArbAgent(BaseAgent):
    def run(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        series_a = data.get("asset_a")
        series_b = data.get("asset_b")

        if series_a is None or series_b is None or len(series_a) != len(series_b):
            return {
                "signal": "neutral",
                "confidence": 0,
                "explanation": "Invalid or unequal asset series"
            }

        score, pvalue, _ = coint(series_a, series_b)

        if pvalue < 0.05:
            spread = series_a - series_b
            zscore = (spread - spread.mean()) / spread.std()
            latest_z = zscore.iloc[-1]

            if latest_z > 1.5:
                return {"signal": "short_a_long_b", "confidence": 0.8, "explanation": f"Z={latest_z:.2f}  mean reversion expected"}
            elif latest_z < -1.5:
                return {"signal": "long_a_short_b", "confidence": 0.8, "explanation": f"Z={latest_z:.2f}  mean reversion expected"}

        return {
            "signal": "neutral",
            "confidence": 0.4,
            "explanation": "No cointegration or z-score extreme"
        }
