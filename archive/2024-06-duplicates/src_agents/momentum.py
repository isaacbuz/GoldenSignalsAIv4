"""
Momentum Trading Agent
Identifies and trades momentum patterns
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent


class MomentumAgent(BaseAgent):
    """Agent that identifies momentum trading opportunities"""
    
    def __init__(self):
        super().__init__("MomentumAgent")
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_threshold = 0
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize momentum agent"""
        if config:
            self.rsi_oversold = config.get('rsi_oversold', 30)
            self.rsi_overbought = config.get('rsi_overbought', 70)
            self.macd_threshold = config.get('macd_threshold', 0)
        
        self.logger.info(f"Momentum Agent initialized with RSI levels: {self.rsi_oversold}/{self.rsi_overbought}")
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data for momentum signals"""
        indicators = market_data.get('indicators', {})
        
        rsi = indicators.get('rsi', 50)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        # Check for momentum signals
        if rsi < self.rsi_oversold and macd_histogram > self.macd_threshold:
            return {
                'type': 'BUY',
                'confidence': min(85, 60 + (self.rsi_oversold - rsi)),
                'strategy': 'MOMENTUM',
                'reasoning': [
                    f"RSI oversold at {rsi:.1f}",
                    "MACD showing bullish momentum"
                ]
            }
        elif rsi > self.rsi_overbought and macd_histogram < -self.macd_threshold:
            return {
                'type': 'SELL',
                'confidence': min(85, 60 + (rsi - self.rsi_overbought)),
                'strategy': 'MOMENTUM',
                'reasoning': [
                    f"RSI overbought at {rsi:.1f}",
                    "MACD showing bearish momentum"
                ]
            }
        
        return None 