from starlette.concurrency import run_in_threadpool
from typing import Any, Dict, List

from agents.base import BaseAgent, AgentConfig
from src.models.signals import Signal
from src.utils.legacy_mapper import legacy_output_to_signal

# Import the original legacy RSI agent from the old code tree
from agents.core.technical.momentum.rsi_agent import RSIAgent as LegacyRSIAgent


class RSILegacyAdapter(BaseAgent):
    """Async adapter around the legacy RSI technical strategy.

    It forwards incoming market data to the legacy agent, then translates
    its synchronous dict response into the modern :class:`~src.models.signals.Signal`.
    """

    def __init__(self, db_manager, redis_manager):
        cfg = AgentConfig(
            name="legacy_rsi",
            version="legacy",
            weight=0.10,
            confidence_threshold=0.0,  # let orchestrator decide
        )
        super().__init__(config=cfg, db_manager=db_manager, redis_manager=redis_manager)
        self._legacy = LegacyRSIAgent()

    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> Signal:
        """Run RSI calculation in a threadpool and return a V3 Signal."""
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

    # ------------------------------------------------------------------
    # Metadata required by the orchestrator
    # ------------------------------------------------------------------
    def get_required_data_types(self):
        return ["ohlcv"] 