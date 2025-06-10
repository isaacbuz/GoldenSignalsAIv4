"""
Cross-exchange arbitrage agent implementation.
"""
from typing import Dict, Any, List, Callable, Optional
import logging
import asyncio
from datetime import datetime
from .base import BaseArbitrageAgent, ArbitrageOpportunity

logger = logging.getLogger(__name__)

class CrossExchangeArbitrageAgent(BaseArbitrageAgent):
    """Agent that finds arbitrage opportunities across different exchanges."""
    
    def __init__(
        self,
        name: str = "CrossExchangeArbitrage",
        min_spread: float = 0.01,
        min_volume: float = 100.0,
        max_slippage: float = 0.002,
        max_latency_ms: int = 200,
        fee_rate: float = 0.001,
        data_sources: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize cross-exchange arbitrage agent.
        
        Args:
            name: Agent name
            min_spread: Minimum spread to consider
            min_volume: Minimum volume for opportunities
            max_slippage: Maximum acceptable slippage
            max_latency_ms: Maximum acceptable latency
            fee_rate: Trading fee rate
            data_sources: Dict mapping venue names to price fetching functions
        """
        super().__init__(
            name=name,
            min_spread=min_spread,
            min_volume=min_volume,
            max_slippage=max_slippage,
            max_latency_ms=max_latency_ms,
            fee_rate=fee_rate
        )
        self.data_sources = data_sources or {}
        
    async def fetch_prices(
        self,
        symbol: str,
        venues: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Fetch prices from multiple venues asynchronously."""
        prices = {}
        venues = venues or list(self.data_sources.keys())
        
        async def fetch_single_price(venue: str) -> None:
            try:
                fetcher = self.data_sources[venue]
                if asyncio.iscoroutinefunction(fetcher):
                    price = await fetcher(symbol)
                else:
                    price = fetcher(symbol)
                    
                if price is not None and price > 0:
                    prices[venue] = price
                    
            except Exception as e:
                logger.warning(f"Failed to fetch price from {venue} for {symbol}: {str(e)}")
                
        # Create tasks for all venues
        tasks = [fetch_single_price(venue) for venue in venues]
        await asyncio.gather(*tasks)
        
        return prices
        
    def find_opportunities(
        self,
        symbol: str,
        prices: Dict[str, float]
    ) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities in price data."""
        opportunities = []
        venues = list(prices.keys())
        
        for i in range(len(venues)):
            for j in range(len(venues)):
                if i == j:
                    continue
                    
                buy_venue = venues[i]
                sell_venue = venues[j]
                buy_price = prices[buy_venue]
                sell_price = prices[sell_venue]
                
                # Create opportunity
                opp = ArbitrageOpportunity(
                    symbol=symbol,
                    buy_venue=buy_venue,
                    sell_venue=sell_venue,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    timestamp=datetime.now().timestamp()
                )
                
                # Validate opportunity
                if self.validate_opportunity(opp):
                    opportunities.append(opp)
                    
        return opportunities
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data for arbitrage opportunities."""
        try:
            if "symbol" not in data:
                raise ValueError("Missing symbol in data")
                
            symbol = data["symbol"]
            venues = data.get("venues")  # Optional venue filter
            
            # Get event loop or create new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Fetch prices
            prices = loop.run_until_complete(
                self.fetch_prices(symbol, venues)
            )
            
            if len(prices) < 2:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Insufficient price data",
                        "prices": prices
                    }
                }
                
            # Find opportunities
            opportunities = self.find_opportunities(symbol, prices)
            self.opportunities = opportunities
            
            if not opportunities:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "prices": prices,
                        "opportunities": []
                    }
                }
                
            # Get best opportunity
            best_opp = max(opportunities, key=lambda x: x.spread)
            confidence = min(best_opp.spread / (best_opp.buy_price * self.min_spread), 1.0)
            
            return {
                "action": "execute",
                "confidence": confidence,
                "metadata": {
                    "prices": prices,
                    "opportunities": [opp.to_dict() for opp in opportunities],
                    "best_opportunity": best_opp.to_dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Cross-exchange arbitrage processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 