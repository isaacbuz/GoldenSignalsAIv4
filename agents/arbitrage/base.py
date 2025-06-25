"""
Base arbitrage agent defining common functionality.
"""
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from src.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ArbitrageOpportunity:
    """Represents an arbitrage opportunity with execution details."""
    
    def __init__(
        self,
        symbol: str,
        buy_venue: str,
        sell_venue: str,
        buy_price: float,
        sell_price: float,
        volume: Optional[float] = None,
        timestamp: Optional[float] = None
    ):
        self.symbol = symbol
        self.buy_venue = buy_venue
        self.sell_venue = sell_venue
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.spread = sell_price - buy_price
        self.volume = volume
        self.timestamp = timestamp or datetime.now().timestamp()
        self.status = 'Open'  # Open, Executed, Missed, Failed
        self.execution_details: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary format."""
        return {
            'symbol': self.symbol,
            'buy_venue': self.buy_venue,
            'sell_venue': self.sell_venue,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'spread': self.spread,
            'volume': self.volume,
            'timestamp': self.timestamp,
            'status': self.status,
            'execution_details': self.execution_details
        }

class BaseArbitrageAgent(BaseAgent):
    """Base class for arbitrage agents."""
    
    def __init__(
        self,
        name: str,
        min_spread: float = 0.01,
        min_volume: float = 100.0,
        max_slippage: float = 0.002,
        max_latency_ms: int = 200,
        fee_rate: float = 0.001
    ):
        """
        Initialize base arbitrage agent.
        
        Args:
            name: Agent name
            min_spread: Minimum spread to consider (as fraction)
            min_volume: Minimum volume for opportunities
            max_slippage: Maximum acceptable slippage
            max_latency_ms: Maximum acceptable latency
            fee_rate: Trading fee rate
        """
        super().__init__(name=name, agent_type="arbitrage")
        self.min_spread = min_spread
        self.min_volume = min_volume
        self.max_slippage = max_slippage
        self.max_latency_ms = max_latency_ms
        self.fee_rate = fee_rate
        self.opportunities: List[ArbitrageOpportunity] = []
        
    def is_profitable(self, opportunity: ArbitrageOpportunity) -> bool:
        """Check if opportunity is profitable after fees and slippage."""
        try:
            # Calculate total fees
            volume = opportunity.volume or self.min_volume
            total_fee = (opportunity.buy_price + opportunity.sell_price) * volume * self.fee_rate
            
            # Estimate worst-case slippage
            slippage_cost = (opportunity.buy_price + opportunity.sell_price) * volume * self.max_slippage
            
            # Calculate net profit
            gross_profit = opportunity.spread * volume
            net_profit = gross_profit - total_fee - slippage_cost
            
            return net_profit > 0
            
        except Exception as e:
            logger.error(f"Profitability check failed: {str(e)}")
            return False
            
    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate opportunity meets basic requirements."""
        try:
            if not opportunity.buy_price > 0 or not opportunity.sell_price > 0:
                return False
                
            if opportunity.spread < self.min_spread:
                return False
                
            if opportunity.volume and opportunity.volume < self.min_volume:
                return False
                
            return self.is_profitable(opportunity)
            
        except Exception as e:
            logger.error(f"Opportunity validation failed: {str(e)}")
            return False
            
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data for arbitrage opportunities.
        To be implemented by specific arbitrage strategies.
        """
        raise NotImplementedError("Subclasses must implement process()")
        
    def get_opportunities(self) -> List[Dict[str, Any]]:
        """Get current opportunities in dictionary format."""
        return [opp.to_dict() for opp in self.opportunities]
        
    def clear_opportunities(self) -> None:
        """Clear current opportunities list."""
        self.opportunities = [] 