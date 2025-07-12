"""Mean Reversion Agent."""

from agents.base_agent import BaseAgent
from typing import Dict, Any, List
from src.ml.models.market_data import MarketData
from src.ml.models.signals import Signal, SignalType, SignalStrength, SignalSource

class ReversionAgent(BaseAgent):
    """Agent that trades mean reversion strategies."""
    
    def __init__(self, name: str = "Mean Reversion", lookback: int = 20):
        super().__init__(name=name, agent_type="technical")
        self.lookback = lookback
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for mean reversion signals."""
        return {
            "action": "hold",
            "confidence": 0.5,
            "metadata": {"deviation": 0.0}
        }
    
    async def analyze(self, market_data: MarketData) -> Signal:
        """Analyze for mean reversion opportunities."""
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
        return ["close_prices", "ohlcv"]
