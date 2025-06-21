"""
Technical Analysis Trading Agent
Analyzes technical patterns and trends
"""

from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent


class TechnicalAnalysisAgent(BaseAgent):
    """Agent that performs technical analysis"""
    
    def __init__(self):
        super().__init__("TechnicalAnalysisAgent")
        self.volume_threshold = 1.5
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize technical analysis agent"""
        if config:
            self.volume_threshold = config.get('volume_threshold', 1.5)
        
        self.logger.info(f"Technical Analysis Agent initialized")
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data for technical patterns"""
        indicators = market_data.get('indicators', {})
        
        trend = indicators.get('trend', 'neutral')
        volume_ratio = indicators.get('volume_ratio', 1)
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        
        # Check for technical signals
        if trend == 'bullish' and volume_ratio > self.volume_threshold:
            confidence = min(85, 70 + (volume_ratio - 1.5) * 20)
            return {
                'type': 'BUY',
                'confidence': confidence,
                'strategy': 'TECHNICAL',
                'reasoning': [
                    "Bullish trend confirmed",
                    f"High volume ratio: {volume_ratio:.1f}x average"
                ]
            }
        elif trend == 'bearish' and volume_ratio > self.volume_threshold:
            confidence = min(85, 70 + (volume_ratio - 1.5) * 20)
            return {
                'type': 'SELL',
                'confidence': confidence,
                'strategy': 'TECHNICAL',
                'reasoning': [
                    "Bearish trend confirmed",
                    f"High volume ratio: {volume_ratio:.1f}x average"
                ]
            }
        
        # Check for moving average crossovers
        if sma_20 and sma_50:
            if sma_20 > sma_50 * 1.02:  # Golden cross
                return {
                    'type': 'BUY',
                    'confidence': 75,
                    'strategy': 'TECHNICAL',
                    'reasoning': [
                        "Golden cross pattern detected",
                        "20 SMA crossed above 50 SMA"
                    ]
                }
            elif sma_20 < sma_50 * 0.98:  # Death cross
                return {
                    'type': 'SELL',
                    'confidence': 75,
                    'strategy': 'TECHNICAL',
                    'reasoning': [
                        "Death cross pattern detected",
                        "20 SMA crossed below 50 SMA"
                    ]
                }
        
        return None 