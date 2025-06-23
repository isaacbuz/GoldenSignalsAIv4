from starlette.concurrency import run_in_threadpool
from typing import Any, Dict, List

import pandas as pd

from agents.base import BaseAgent, AgentConfig
from src.utils.legacy_mapper import legacy_output_to_signal
from src.models.signals import Signal

from archive.legacy_backend_agents.volume.obv_agent import OBVAgent as LegacyOBVAgent


class OBVLegacyAdapter(BaseAgent):
    """Adapter for legacy On-Balance Volume agent."""

    def __init__(self, db_manager, redis_manager):
        cfg = AgentConfig(
            name="legacy_obv",
            version="legacy",
            weight=0.08,
            confidence_threshold=0.0,
        )
        super().__init__(config=cfg, db_manager=db_manager, redis_manager=redis_manager)
        self._legacy = LegacyOBVAgent()

    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> Signal:
        # Build DataFrame with close and volume
        df = pd.DataFrame([
            {"close": r.get("close"), "volume": r.get("volume", 0)} for r in historical_data
        ])

        legacy_output = await run_in_threadpool(self._legacy.run, df)

        # Map bullish/bearish/neutral to buy/sell/hold
        action_map = {
            "bullish": "buy",
            "bearish": "sell",
            "neutral": "hold",
        }
        legacy_output["action"] = action_map.get(legacy_output.get("signal", "hold"), "hold")

        return legacy_output_to_signal(
            legacy_output,
            symbol=symbol,
            current_price=market_data.get("price"),
            source=self.config.name,
        )

    def get_required_data_types(self):
        return ["ohlcv"] 