"""
Portfolio management module with advanced risk management features.
Inspired by AlphaPy's portfolio management approach.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(
        self,
        initial_capital: float,
        max_position_size: float = 0.2,
        max_risk_per_trade: float = 0.02,
        volatility_lookback: int = 20
    ):
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.volatility_lookback = volatility_lookback
        self.positions: Dict[str, Dict] = {}
        
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        volatility: float,
        signal_confidence: float
    ) -> Dict[str, float]:
        """Calculate position size based on volatility and signal confidence."""
        try:
            # Base position size as percentage of capital
            base_size = self.capital * self.max_position_size
            
            # Adjust for volatility
            vol_factor = 1.0 / (1.0 + volatility)
            
            # Adjust for signal confidence
            conf_factor = signal_confidence
            
            # Calculate final position size
            position_size = base_size * vol_factor * conf_factor
            
            # Ensure we don't exceed max risk
            max_risk_amount = self.capital * self.max_risk_per_trade
            stop_loss = price * (1 - volatility)  # Simple volatility-based stop loss
            max_position = max_risk_amount / (price - stop_loss) if price > stop_loss else 0
            
            position_size = min(position_size, max_position)
            
            return {
                "size": position_size,
                "stop_loss": stop_loss,
                "risk_amount": position_size * (price - stop_loss)
            }
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            return {"size": 0, "stop_loss": 0, "risk_amount": 0}
            
    def adjust_for_correlation(
        self,
        symbol: str,
        position_size: float,
        correlation_matrix: pd.DataFrame
    ) -> float:
        """Adjust position size based on portfolio correlation."""
        try:
            if symbol in correlation_matrix.columns:
                # Calculate average correlation with existing positions
                correlations = []
                for pos in self.positions:
                    if pos in correlation_matrix.columns:
                        correlations.append(abs(correlation_matrix.loc[symbol, pos]))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    # Reduce position size for high correlations
                    return position_size * (1 - avg_correlation * 0.5)
                    
            return position_size
            
        except Exception as e:
            logger.error(f"Correlation adjustment failed: {str(e)}")
            return position_size
            
    def rebalance_portfolio(
        self,
        market_data: Dict[str, pd.DataFrame],
        signals: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Rebalance portfolio based on new signals and market conditions."""
        try:
            rebalance_actions = {}
            total_risk = 0
            
            # Calculate portfolio value and risk
            portfolio_value = self.capital
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    current_price = market_data[symbol]['close'].iloc[-1]
                    position_value = position['size'] * current_price
                    portfolio_value += position_value
                    total_risk += position['risk_amount']
            
            # Process new signals
            for symbol, signal in signals.items():
                if symbol not in market_data:
                    continue
                    
                price_data = market_data[symbol]
                current_price = price_data['close'].iloc[-1]
                volatility = price_data['close'].pct_change().std() * np.sqrt(252)
                
                # Calculate new position size
                position_info = self.calculate_position_size(
                    symbol,
                    current_price,
                    volatility,
                    signal.get('confidence', 0.5)
                )
                
                # Adjust for correlations if we have multiple positions
                if len(self.positions) > 1:
                    correlation_matrix = pd.DataFrame({
                        sym: data['close'] for sym, data in market_data.items()
                    }).corr()
                    position_info['size'] = self.adjust_for_correlation(
                        symbol,
                        position_info['size'],
                        correlation_matrix
                    )
                
                # Determine action based on signal and current position
                if signal['action'] == 'buy' and symbol not in self.positions:
                    if total_risk + position_info['risk_amount'] <= portfolio_value * self.max_risk_per_trade:
                        rebalance_actions[symbol] = {
                            'action': 'buy',
                            'size': position_info['size'],
                            'stop_loss': position_info['stop_loss']
                        }
                elif signal['action'] == 'sell' and symbol in self.positions:
                    rebalance_actions[symbol] = {
                        'action': 'sell',
                        'size': self.positions[symbol]['size']
                    }
                    
            return rebalance_actions
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {str(e)}")
            return {} 