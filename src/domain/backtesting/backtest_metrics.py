"""
Backtesting Metrics Module - Handles performance metrics calculation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.backtesting.advanced_backtest_engine import BacktestMetrics, BacktestTrade

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates comprehensive backtesting metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
        
    def calculate_metrics(
        self,
        trades: List[BacktestTrade],
        initial_capital: float,
        equity_curve: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> BacktestMetrics:
        """Calculate all backtesting metrics"""
        
        if not trades:
            return self._empty_metrics()
        
        # Basic trade statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Returns and risk metrics
        returns = equity_curve.pct_change().dropna()
        risk_metrics = self._calculate_risk_metrics(returns, equity_curve)
        
        # Benchmark relative metrics
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(
                returns, benchmark_returns
            )
        
        # Combine all metrics
        metrics = BacktestMetrics(
            # Returns
            total_return=self._calculate_total_return(equity_curve, initial_capital),
            annualized_return=self._annualize_return(returns),
            benchmark_return=benchmark_metrics.get('benchmark_return', 0),
            alpha=benchmark_metrics.get('alpha', 0),
            beta=benchmark_metrics.get('beta', 0),
            
            # Risk
            volatility=self._calculate_volatility(returns),
            downside_volatility=self._calculate_downside_volatility(returns),
            max_drawdown=risk_metrics['max_drawdown'],
            max_drawdown_duration=risk_metrics['max_drawdown_duration'],
            var_95=self._calculate_var(returns, 0.95),
            cvar_95=self._calculate_cvar(returns, 0.95),
            
            # Risk-adjusted returns
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            sortino_ratio=self._calculate_sortino_ratio(returns),
            calmar_ratio=self._calculate_calmar_ratio(
                self._annualize_return(returns),
                risk_metrics['max_drawdown']
            ),
            information_ratio=benchmark_metrics.get('information_ratio', 0),
            
            # Trade statistics
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            expectancy=trade_stats['expectancy'],
            payoff_ratio=trade_stats['payoff_ratio'],
            
            # Timing
            avg_trade_duration=trade_stats['avg_trade_duration'],
            trades_per_day=self._calculate_trades_per_day(trades),
            exposure_time=self._calculate_exposure_time(trades, equity_curve)
        )
        
        return metrics
    
    def _calculate_trade_statistics(self, trades: List[BacktestTrade]) -> Dict:
        """Calculate trade-level statistics"""
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': num_wins / total_trades if total_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'expectancy': (avg_win * num_wins - avg_loss * num_losses) / total_trades if total_trades > 0 else 0,
            'payoff_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'avg_trade_duration': self._calculate_avg_duration(trades)
        }
    
    def _calculate_risk_metrics(
        self, 
        returns: pd.Series, 
        equity_curve: pd.Series
    ) -> Dict:
        """Calculate risk-related metrics"""
        # Drawdown calculation
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        max_drawdown = abs(drawdown.min())
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for date, dd in drawdown.items():
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = date
                current_duration = (date - drawdown_start).days
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_start = None
                current_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_duration,
            'drawdown_series': drawdown
        }
    
    def _calculate_benchmark_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate benchmark-relative metrics"""
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )
        
        if len(aligned_returns) == 0:
            return {}
        
        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (annualized)
        portfolio_return = self._annualize_return(aligned_returns)
        benchmark_return = self._annualize_return(aligned_benchmark)
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Information ratio
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(self.trading_days_per_year)
        information_ratio = (
            active_returns.mean() * self.trading_days_per_year / tracking_error
            if tracking_error > 0 else 0
        )
        
        return {
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio
        }
    
    def _calculate_total_return(
        self, 
        equity_curve: pd.Series, 
        initial_capital: float
    ) -> float:
        """Calculate total return"""
        final_value = equity_curve.iloc[-1]
        return (final_value - initial_capital) / initial_capital
    
    def _annualize_return(self, returns: pd.Series) -> float:
        """Annualize returns"""
        if len(returns) == 0:
            return 0
        
        total_days = (returns.index[-1] - returns.index[0]).days
        if total_days == 0:
            return 0
            
        years = total_days / 365.25
        cumulative_return = (1 + returns).prod() - 1
        
        return (1 + cumulative_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(self.trading_days_per_year)
    
    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """Calculate downside volatility (for Sortino ratio)"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        return downside_returns.std() * np.sqrt(self.trading_days_per_year)
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        volatility = returns.std()
        
        if volatility == 0:
            return 0
            
        return excess_returns.mean() / volatility * np.sqrt(self.trading_days_per_year)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        downside_vol = self._calculate_downside_volatility(returns) / np.sqrt(self.trading_days_per_year)
        
        if downside_vol == 0:
            return 0
            
        return excess_returns.mean() / downside_vol * np.sqrt(self.trading_days_per_year)
    
    def _calculate_calmar_ratio(
        self, 
        annualized_return: float, 
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0
        return annualized_return / max_drawdown
    
    def _calculate_avg_duration(self, trades: List[BacktestTrade]) -> timedelta:
        """Calculate average trade duration"""
        if not trades:
            return timedelta(0)
            
        durations = [t.time_in_trade for t in trades]
        avg_seconds = np.mean([d.total_seconds() for d in durations])
        return timedelta(seconds=avg_seconds)
    
    def _calculate_trades_per_day(self, trades: List[BacktestTrade]) -> float:
        """Calculate average trades per day"""
        if not trades:
            return 0
            
        first_trade = min(t.entry_time for t in trades)
        last_trade = max(t.exit_time for t in trades)
        total_days = (last_trade - first_trade).days
        
        if total_days == 0:
            return len(trades)
            
        return len(trades) / total_days
    
    def _calculate_exposure_time(
        self, 
        trades: List[BacktestTrade], 
        equity_curve: pd.Series
    ) -> float:
        """Calculate percentage of time in market"""
        if not trades or len(equity_curve) < 2:
            return 0
            
        total_time = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
        if total_time == 0:
            return 0
            
        time_in_market = sum(
            t.time_in_trade.total_seconds() for t in trades
        )
        
        return time_in_market / total_time
    
    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics when no trades"""
        return BacktestMetrics(
            total_return=0,
            annualized_return=0,
            benchmark_return=0,
            alpha=0,
            beta=0,
            volatility=0,
            downside_volatility=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            var_95=0,
            cvar_95=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            information_ratio=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            expectancy=0,
            payoff_ratio=0,
            avg_trade_duration=timedelta(0),
            trades_per_day=0,
            exposure_time=0
        ) 