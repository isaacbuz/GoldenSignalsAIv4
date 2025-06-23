"""
Metrics Collection Utilities - GoldenSignalsAI V3

Utilities for collecting and tracking performance metrics.
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Utility class for collecting and tracking performance metrics
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            max_history: Maximum number of historical data points to keep
        """
        self.max_history = max_history
        self.metrics = defaultdict(deque)
        self.counters = defaultdict(int)
        self.timers = {}
        self.start_time = time.time()
        
    def increment(self, metric_name: str, value: int = 1) -> None:
        """
        Increment a counter metric
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        self.counters[metric_name] += value
        
    def record(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record a metric value with timestamp
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            timestamp: Timestamp for the metric (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        metric_queue = self.metrics[metric_name]
        metric_queue.append((timestamp, value))
        
        # Keep only the most recent entries
        while len(metric_queue) > self.max_history:
            metric_queue.popleft()
            
    def start_timer(self, timer_name: str) -> None:
        """
        Start a timer for measuring duration
        
        Args:
            timer_name: Name of the timer
        """
        self.timers[timer_name] = time.time()
        
    def stop_timer(self, timer_name: str) -> Optional[float]:
        """
        Stop a timer and record the duration
        
        Args:
            timer_name: Name of the timer
            
        Returns:
            Duration in seconds, or None if timer wasn't started
        """
        if timer_name not in self.timers:
            logger.warning(f"Timer '{timer_name}' was not started")
            return None
            
        start_time = self.timers.pop(timer_name)
        duration = time.time() - start_time
        
        # Record the duration as a metric
        self.record(f"{timer_name}_duration", duration)
        
        return duration
        
    def get_counter(self, metric_name: str) -> int:
        """
        Get current value of a counter metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Current counter value
        """
        return self.counters[metric_name]
        
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get the latest value of a metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest metric value, or None if no data
        """
        metric_queue = self.metrics[metric_name]
        if metric_queue:
            return metric_queue[-1][1]
        return None
        
    def get_average(self, metric_name: str, window_minutes: Optional[int] = None) -> Optional[float]:
        """
        Get average value of a metric over a time window
        
        Args:
            metric_name: Name of the metric
            window_minutes: Time window in minutes (None for all data)
            
        Returns:
            Average value, or None if no data
        """
        metric_queue = self.metrics[metric_name]
        if not metric_queue:
            return None
            
        if window_minutes is None:
            # Use all data
            values = [value for _, value in metric_queue]
        else:
            # Filter by time window
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            values = [value for timestamp, value in metric_queue if timestamp >= cutoff_time]
            
        if not values:
            return None
            
        return sum(values) / len(values)
        
    def get_rate(self, metric_name: str, window_minutes: int = 1) -> float:
        """
        Get rate of a counter metric (events per minute)
        
        Args:
            metric_name: Name of the counter metric
            window_minutes: Time window in minutes
            
        Returns:
            Rate in events per minute
        """
        current_count = self.counters[metric_name]
        elapsed_minutes = (time.time() - self.start_time) / 60
        
        if elapsed_minutes == 0:
            return 0.0
            
        return current_count / min(elapsed_minutes, window_minutes)
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        
        Returns:
            Dictionary containing metric summaries
        """
        summary = {
            "counters": dict(self.counters),
            "metrics": {},
            "uptime_seconds": time.time() - self.start_time
        }
        
        for metric_name, metric_queue in self.metrics.items():
            if metric_queue:
                values = [value for _, value in metric_queue]
                summary["metrics"][metric_name] = {
                    "latest": values[-1],
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
                
        return summary
        
    def reset(self) -> None:
        """Reset all metrics and counters"""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        self.start_time = time.time()
        
    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """
        Clean up old metric data
        
        Args:
            max_age_hours: Maximum age of data to keep in hours
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for metric_name, metric_queue in self.metrics.items():
            # Remove old entries
            while metric_queue and metric_queue[0][0] < cutoff_time:
                metric_queue.popleft()


# Global metrics collector instance
metrics = MetricsCollector()


# Metric calculation functions
def calculate_win_rate(wins: int, losses: int) -> float:
    """
    Calculate win rate from wins and losses
    
    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        
    Returns:
        Win rate as a decimal (0-1)
    """
    total_trades = wins + losses
    if total_trades == 0:
        return 0.0
    return wins / total_trades


def calculate_profit_factor(profits: list, losses: list) -> float:
    """
    Calculate profit factor (sum of profits / sum of losses)
    
    Args:
        profits: List of profit amounts
        losses: List of loss amounts (as positive values)
        
    Returns:
        Profit factor ratio
    """
    total_profits = sum(profits) if profits else 0
    total_losses = sum(losses) if losses else 0
    
    if total_losses == 0:
        return float('inf') if total_profits > 0 else 0.0
    
    return total_profits / total_losses


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess == 0:
        return float('inf') if mean_excess > 0 else float('-inf') if mean_excess < 0 else 0.0
    
    return mean_excess / std_excess * np.sqrt(252)  # Annualized 