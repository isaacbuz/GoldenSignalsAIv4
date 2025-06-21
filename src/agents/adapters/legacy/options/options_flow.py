from starlette.concurrency import run_in_threadpool
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent, AgentConfig
from src.models.signals import Signal
from src.utils.legacy_mapper import legacy_output_to_signal

# Legacy implementation path
from src.agents.research.ml.options_flow import OptionsFlowAgent as LegacyOptionsFlowAgent


class OptionsFlowLegacyAdapter(BaseAgent):
    """Adapter for the legacy Options Flow strategy.

    The legacy agent expects a dict with key ``options_data`` that contains a
    list of option trade dictionaries.  We pluck that from the incoming
    `market_data` dict.  If no option data is present we skip analysis and
    return ``None`` so the orchestrator will treat it as a non-participant in
    this cycle.
    """

    def __init__(self, db_manager, redis_manager):
        cfg = AgentConfig(
            name="legacy_options_flow",
            version="legacy",
            weight=0.07,
            confidence_threshold=0.0,
        )
        super().__init__(config=cfg, db_manager=db_manager, redis_manager=redis_manager)
        self._legacy = LegacyOptionsFlowAgent()

    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],  # unused for this agent
    ) -> Optional[Signal]:
        # Retrieve options data the orchestrator may have placed in market_data
        options_data = market_data.get("options_data")
        if not options_data:
            # No data → skip this cycle
            return None

        legacy_output = await run_in_threadpool(
            self._legacy.process,
            {"options_data": options_data},
        )

        return legacy_output_to_signal(
            legacy_output,
            symbol=symbol,
            current_price=market_data.get("price"),
            source=self.config.name,
        )

    def get_required_data_types(self):
        return ["options_flow"] 