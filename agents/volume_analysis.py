"""
Volume Analysis Trading Agent
Analyzes trading volume patterns
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent


class VolumeAnalysisAgent(BaseAgent):
    """Agent that analyzes trading volume"""
    
    def __init__(self):
        super().__init__("VolumeAnalysisAgent")
        self.volume_surge_threshold = 2.0
        self.price_change_threshold = 0.02
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize volume analysis agent"""
        if config:
            self.volume_surge_threshold = config.get('volume_surge_threshold', 2.0)
            self.price_change_threshold = config.get('price_change_threshold', 0.02)
        
        self.logger.info(f"Volume Analysis Agent initialized")
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data for volume patterns"""
        indicators = market_data.get('indicators', {})
        historical = market_data.get('historical', [])
        
        volume_ratio = indicators.get('volume_ratio', 1)
        
        if volume_ratio > self.volume_surge_threshold and len(historical) >= 2:
            # Check price action with volume
            price_change = (historical[-1]['close'] - historical[-2]['close']) / historical[-2]['close']
            
            if price_change > self.price_change_threshold:
                return {
                    'type': 'BUY',
                    'confidence': min(85, 70 + (volume_ratio - 2) * 10),
                    'strategy': 'VOLUME',
                    'reasoning': [
                        f"Volume surge {volume_ratio:.1f}x average",
                        "Strong buying pressure detected"
                    ]
                }
            elif price_change < -self.price_change_threshold:
                return {
                    'type': 'SELL',
                    'confidence': min(85, 70 + (volume_ratio - 2) * 10),
                    'strategy': 'VOLUME',
                    'reasoning': [
                        f"Volume surge {volume_ratio:.1f}x average",
                        "Strong selling pressure detected"
                    ]
                }
        
        return None 