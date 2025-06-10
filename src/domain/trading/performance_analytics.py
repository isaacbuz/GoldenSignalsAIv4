"""
Performance analytics module with advanced metrics.
Inspired by AlphaPy's performance analysis features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive return metrics."""
        try:
            metrics = {}
            
            # Basic return metrics
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
            metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Risk metrics
            metrics['sharpe_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / metrics['volatility']
            metrics['sortino_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252))
            
            # Drawdown analysis
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdowns.min()
            metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean()
            metrics['drawdown_duration'] = self._calculate_avg_drawdown_duration(drawdowns)
            
            # Trading metrics
            metrics['win_rate'] = len(returns[returns > 0]) / len(returns)
            metrics['profit_factor'] = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
            metrics['avg_win'] = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            metrics['avg_loss'] = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Returns metrics calculation failed: {str(e)}")
            return {}
            
    def _calculate_avg_drawdown_duration(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration in days."""
        try:
            # Find drawdown periods
            is_drawdown = drawdowns < 0
            drawdown_starts = is_drawdown.astype(int).diff()
            
            # Calculate durations
            durations = []
            current_duration = 0
            
            for i in range(len(drawdowns)):
                if drawdowns[i] < 0:
                    current_duration += 1
                elif current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
                    
            if current_duration > 0:
                durations.append(current_duration)
                
            return np.mean(durations) if durations else 0
            
        except Exception as e:
            logger.error(f"Drawdown duration calculation failed: {str(e)}")
            return 0
            
    def analyze_trades(self, trades: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze individual trade performance."""
        try:
            analysis = {
                'overall': {},
                'by_symbol': {},
                'by_signal': {}
            }
            
            # Overall trade metrics
            analysis['overall'] = {
                'total_trades': len(trades),
                'profitable_trades': len(trades[trades['pnl'] > 0]),
                'win_rate': len(trades[trades['pnl'] > 0]) / len(trades),
                'avg_profit': trades[trades['pnl'] > 0]['pnl'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0,
                'avg_loss': trades[trades['pnl'] < 0]['pnl'].mean() if len(trades[trades['pnl'] < 0]) > 0 else 0,
                'largest_win': trades['pnl'].max(),
                'largest_loss': trades['pnl'].min(),
                'avg_hold_time': (trades['exit_time'] - trades['entry_time']).mean().total_seconds() / 86400  # in days
            }
            
            # Analysis by symbol
            for symbol in trades['symbol'].unique():
                symbol_trades = trades[trades['symbol'] == symbol]
                analysis['by_symbol'][symbol] = {
                    'total_trades': len(symbol_trades),
                    'win_rate': len(symbol_trades[symbol_trades['pnl'] > 0]) / len(symbol_trades),
                    'avg_profit': symbol_trades[symbol_trades['pnl'] > 0]['pnl'].mean() if len(symbol_trades[symbol_trades['pnl'] > 0]) > 0 else 0,
                    'total_pnl': symbol_trades['pnl'].sum()
                }
                
            # Analysis by signal type
            for signal in trades['signal_type'].unique():
                signal_trades = trades[trades['signal_type'] == signal]
                analysis['by_signal'][signal] = {
                    'total_trades': len(signal_trades),
                    'win_rate': len(signal_trades[signal_trades['pnl'] > 0]) / len(signal_trades),
                    'avg_profit': signal_trades[signal_trades['pnl'] > 0]['pnl'].mean() if len(signal_trades[signal_trades['pnl'] > 0]) > 0 else 0,
                    'total_pnl': signal_trades['pnl'].sum()
                }
                
            return analysis
            
        except Exception as e:
            logger.error(f"Trade analysis failed: {str(e)}")
            return {} 