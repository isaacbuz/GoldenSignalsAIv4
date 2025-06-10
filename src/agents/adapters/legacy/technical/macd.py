from starlette.concurrency import run_in_threadpool
from typing import Any, Dict, List

from src.agents.base import BaseAgent, AgentConfig
from src.utils.legacy_mapper import legacy_output_to_signal
from src.models.signals import Signal

from agents.core.technical.momentum.macd_agent import MACDAgent as LegacyMACDAgent


class MACDLegacyAdapter(BaseAgent):
    """Adapter for legacy MACD strategy."""

    def __init__(self, db_manager, redis_manager):
        cfg = AgentConfig(
            name="legacy_macd",
            version="legacy",
            weight=0.10,
            confidence_threshold=0.0,
        )
        super().__init__(config=cfg, db_manager=db_manager, redis_manager=redis_manager)
        self._legacy = LegacyMACDAgent()

    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> Signal:
        close_prices = [c.get("close") for c in historical_data if c.get("close") is not None]
        legacy_output = await run_in_threadpool(
            self._legacy.process,
            {"close_prices": close_prices},
        )
        return legacy_output_to_signal(
            legacy_output,
            symbol=symbol,
            current_price=market_data.get("price"),
            source=self.config.name,
        )

    def get_required_data_types(self):
        return ["ohlcv"] 