"""
Arbitrage execution module for simulating and executing trades.
"""
from typing import Dict, Any, Optional, List
import logging
import random
from datetime import datetime
from .base import ArbitrageOpportunity

logger = logging.getLogger(__name__)

class ArbitrageExecutor:
    """Handles execution of arbitrage opportunities."""
    
    def __init__(
        self,
        slippage_model: str = "random",
        base_slippage: float = 0.001,
        max_slippage: float = 0.003,
        min_latency_ms: int = 100,
        max_latency_ms: int = 500,
        min_fill_rate: float = 0.7,
        fee_rate: float = 0.001,
        broker_apis: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize arbitrage executor.
        
        Args:
            slippage_model: Slippage simulation model ('random', 'volume', 'impact')
            base_slippage: Base slippage rate
            max_slippage: Maximum slippage rate
            min_latency_ms: Minimum execution latency
            max_latency_ms: Maximum execution latency
            min_fill_rate: Minimum order fill rate
            fee_rate: Trading fee rate
            broker_apis: Dict mapping venue names to broker API clients
        """
        self.slippage_model = slippage_model
        self.base_slippage = base_slippage
        self.max_slippage = max_slippage
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.min_fill_rate = min_fill_rate
        self.fee_rate = fee_rate
        self.broker_apis = broker_apis or {}
        
    def simulate_slippage(
        self,
        price: float,
        volume: Optional[float] = None,
        market_impact: Optional[float] = None
    ) -> float:
        """Simulate price slippage based on configured model."""
        try:
            if self.slippage_model == "random":
                return random.uniform(
                    self.base_slippage,
                    self.max_slippage
                )
                
            elif self.slippage_model == "volume":
                if volume is None:
                    return self.base_slippage
                # More volume = more slippage
                vol_factor = min(volume / 1000, 1.0)  # Normalize volume
                return self.base_slippage + (
                    self.max_slippage - self.base_slippage
                ) * vol_factor
                
            elif self.slippage_model == "impact":
                if market_impact is None:
                    return self.base_slippage
                # Higher impact = more slippage
                return min(
                    self.base_slippage + market_impact,
                    self.max_slippage
                )
                
            else:
                return self.base_slippage
                
        except Exception as e:
            logger.error(f"Slippage simulation failed: {str(e)}")
            return self.base_slippage
            
    def simulate_latency(self) -> int:
        """Simulate execution latency in milliseconds."""
        return random.randint(
            self.min_latency_ms,
            self.max_latency_ms
        )
        
    def simulate_fill(
        self,
        volume: float,
        spread: float
    ) -> float:
        """Simulate order fill rate."""
        try:
            # Base fill rate on spread (higher spread = higher fill probability)
            base_fill = min(
                1.0,
                max(
                    self.min_fill_rate,
                    spread * 10  # Scale factor
                )
            )
            
            # Add random variation
            variation = random.uniform(-0.1, 0.1)
            fill_rate = max(0.0, min(1.0, base_fill + variation))
            
            return volume * fill_rate
            
        except Exception as e:
            logger.error(f"Fill simulation failed: {str(e)}")
            return volume * self.min_fill_rate
            
    def execute(
        self,
        opportunity: ArbitrageOpportunity,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate execution of arbitrage opportunity.
        
        Args:
            opportunity: The arbitrage opportunity to execute
            market_data: Optional market data for better simulation
        
        Returns:
            Dict containing execution results
        """
        try:
            # Get execution parameters
            volume = opportunity.volume or 100.0  # Default volume
            spread = opportunity.spread
            
            # Simulate execution conditions
            buy_slippage = self.simulate_slippage(
                opportunity.buy_price,
                volume,
                market_data.get("market_impact") if market_data else None
            )
            sell_slippage = self.simulate_slippage(
                opportunity.sell_price,
                volume,
                market_data.get("market_impact") if market_data else None
            )
            
            latency = self.simulate_latency()
            filled_volume = self.simulate_fill(volume, spread)
            
            # Calculate execution prices
            executed_buy = opportunity.buy_price * (1 + buy_slippage)
            executed_sell = opportunity.sell_price * (1 - sell_slippage)
            realized_spread = executed_sell - executed_buy
            
            # Calculate fees
            fees = (executed_buy + executed_sell) * filled_volume * self.fee_rate
            
            # Calculate PnL
            gross_pnl = realized_spread * filled_volume
            net_pnl = gross_pnl - fees
            
            # Update opportunity status
            if filled_volume > 0:
                status = "Executed"
            else:
                status = "Failed"
                
            # Record execution details
            execution_details = {
                "timestamp": datetime.now().timestamp(),
                "status": status,
                "filled_volume": filled_volume,
                "fill_rate": filled_volume / volume,
                "executed_buy": executed_buy,
                "executed_sell": executed_sell,
                "realized_spread": realized_spread,
                "buy_slippage": buy_slippage,
                "sell_slippage": sell_slippage,
                "latency_ms": latency,
                "fees": fees,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl
            }
            
            # Update opportunity
            opportunity.status = status
            opportunity.execution_details = execution_details
            
            return execution_details
            
        except Exception as e:
            logger.error(f"Execution simulation failed: {str(e)}")
            opportunity.status = "Failed"
            opportunity.execution_details = {"error": str(e)}
            return {"error": str(e)}
            
    def execute_live(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: float
    ) -> Dict[str, Any]:
        """
        Execute a live arbitrage trade using broker APIs.
        
        Args:
            opportunity: The arbitrage opportunity to execute
            quantity: Trade quantity
            
        Returns:
            Dict containing execution results
        """
        try:
            # Get broker APIs
            buy_api = self.broker_apis.get(opportunity.buy_venue)
            sell_api = self.broker_apis.get(opportunity.sell_venue)
            
            if not buy_api or not sell_api:
                raise ValueError(f"Missing broker API for {opportunity.buy_venue} or {opportunity.sell_venue}")
                
            # Execute trades
            buy_result = buy_api.buy(
                symbol=opportunity.symbol,
                quantity=quantity,
                price=opportunity.buy_price
            )
            
            sell_result = sell_api.sell(
                symbol=opportunity.symbol,
                quantity=quantity,
                price=opportunity.sell_price
            )
            
            # Calculate execution details
            buy_price = buy_result.get("executed_price", opportunity.buy_price)
            sell_price = sell_result.get("executed_price", opportunity.sell_price)
            filled_quantity = min(
                buy_result.get("filled_quantity", 0),
                sell_result.get("filled_quantity", 0)
            )
            
            # Calculate PnL
            realized_spread = sell_price - buy_price
            fees = (buy_price + sell_price) * filled_quantity * self.fee_rate
            gross_pnl = realized_spread * filled_quantity
            net_pnl = gross_pnl - fees
            
            # Update status
            status = "Executed" if filled_quantity > 0 else "Failed"
            
            # Record execution details
            execution_details = {
                "timestamp": datetime.now().timestamp(),
                "status": status,
                "filled_quantity": filled_quantity,
                "fill_rate": filled_quantity / quantity if quantity > 0 else 0,
                "executed_buy": buy_price,
                "executed_sell": sell_price,
                "realized_spread": realized_spread,
                "fees": fees,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "buy_result": buy_result,
                "sell_result": sell_result
            }
            
            # Update opportunity
            opportunity.status = status
            opportunity.execution_details = execution_details
            
            return execution_details
            
        except Exception as e:
            logger.error(f"Live execution failed: {str(e)}")
            opportunity.status = "Failed"
            opportunity.execution_details = {"error": str(e)}
            return {"error": str(e)}
            
    def execute_batch(
        self,
        opportunities: List[ArbitrageOpportunity],
        quantity: float,
        simulate: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a batch of arbitrage opportunities.
        
        Args:
            opportunities: List of opportunities to execute
            quantity: Trade quantity per opportunity
            simulate: Whether to simulate execution or trade live
            
        Returns:
            Dict containing batch execution results
        """
        results = []
        success_count = 0
        total_pnl = 0.0
        
        for opp in opportunities:
            try:
                # Execute opportunity
                if simulate:
                    result = self.execute(opp)
                else:
                    result = self.execute_live(opp, quantity)
                    
                results.append(result)
                
                # Update statistics
                if result.get("status") == "Executed":
                    success_count += 1
                    total_pnl += result.get("net_pnl", 0)
                    
            except Exception as e:
                logger.error(f"Batch execution failed for {opp.symbol}: {str(e)}")
                results.append({"error": str(e)})
                
        return {
            "success_count": success_count,
            "total_count": len(opportunities),
            "success_rate": success_count / len(opportunities) if opportunities else 0,
            "total_pnl": total_pnl,
            "results": results
        } 