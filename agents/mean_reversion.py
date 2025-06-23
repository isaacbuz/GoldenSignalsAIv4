"""
Mean Reversion Trading Agent
Identifies mean reversion opportunities
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent


class MeanReversionAgent(BaseAgent):
    """Agent that identifies mean reversion trading opportunities"""
    
    def __init__(self):
        super().__init__("MeanReversionAgent")
        self.bb_threshold = 1.0  # How far from bands to trigger
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mean reversion agent"""
        if config:
            self.bb_threshold = config.get('bb_threshold', 1.0)
        
        self.logger.info(f"Mean Reversion Agent initialized")
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data for mean reversion signals"""
        indicators = market_data.get('indicators', {})
        current_price = market_data.get('current_price', 0)
        
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        bb_middle = indicators.get('bb_middle', 0)
        
        if not bb_upper or not bb_lower or not current_price:
            return None
        
        # Check for mean reversion opportunities
        if current_price < bb_lower * self.bb_threshold:
            return {
                'type': 'BUY',
                'confidence': 75,
                'strategy': 'MEAN_REVERSION',
                'reasoning': [
                    "Price below lower Bollinger Band",
                    f"Expecting reversion to mean at {bb_middle:.2f}"
                ]
            }
        elif current_price > bb_upper / self.bb_threshold:
            return {
                'type': 'SELL',
                'confidence': 75,
                'strategy': 'MEAN_REVERSION',
                'reasoning': [
                    "Price above upper Bollinger Band",
                    f"Expecting reversion to mean at {bb_middle:.2f}"
                ]
            }
        
        return None 