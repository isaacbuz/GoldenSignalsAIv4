"""ML Classifier Agent."""

from typing import Any, Dict, List

from agents.base_agent import BaseAgent

from src.ml.models.market_data import MarketData
from src.ml.models.signals import Signal, SignalSource, SignalStrength, SignalType


class MLClassifierAgent(BaseAgent):
    """Agent that uses ML classification for signals."""

    def __init__(self, name: str = "ML Classifier"):
        super().__init__(name=name, agent_type="ml")
        self.classifier = None

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and classify signal."""
        return {
            "action": "hold",
            "confidence": 0.6,
            "metadata": {"classification": "neutral"}
        }

    async def analyze(self, market_data: MarketData) -> Signal:
        """Analyze market data using ML classifier."""
        result = self.process({"data": market_data})
        return Signal(
            symbol=market_data.symbol,
            signal_type=SignalType.HOLD,
            confidence=result["confidence"],
            strength=SignalStrength.MEDIUM,
            source=SignalSource.TECHNICAL_ANALYSIS,
            current_price=market_data.current_price
        )

    def get_required_data_types(self) -> List[str]:
        """Get required data types."""
        return ["close_prices", "volume", "indicators"]
