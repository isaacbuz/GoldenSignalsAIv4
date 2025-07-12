"""LSTM Forecast Agent."""

from agents.base_agent import BaseAgent
from typing import Dict, Any, List
from src.ml.models.market_data import MarketData
from src.ml.models.signals import Signal, SignalType, SignalStrength, SignalSource

class LSTMForecastAgent(BaseAgent):
    """Agent that uses LSTM for price forecasting."""
    
    def __init__(self, name: str = "LSTM Forecast"):
        super().__init__(name=name, agent_type="ml")
        self.model = None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and generate forecast."""
        return {
            "action": "hold",
            "confidence": 0.5,
            "metadata": {"forecast": "neutral"}
        }
    
    async def analyze(self, market_data: MarketData) -> Signal:
        """Analyze market data using LSTM."""
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
        return ["close_prices", "volume", "ohlcv"]
