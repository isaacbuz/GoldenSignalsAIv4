"""
Signal Accuracy Validator - Comprehensive signal quality testing
Tests signal accuracy, tracks false positives/negatives, and validates strategies
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal directions"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class TradingSignal:
    """Represents a trading signal"""
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    confidence: float  # 0-1
    predicted_movement: float  # Expected price change %
    time_horizon: int  # Minutes until expected movement
    agent_source: str
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'predicted_movement': self.predicted_movement,
            'time_horizon': self.time_horizon,
            'agent_source': self.agent_source,
            'features': self.features
        }


@dataclass
class SignalOutcome:
    """Actual outcome of a signal"""
    signal: TradingSignal
    entry_price: float
    exit_price: float
    exit_time: datetime
    actual_movement: float  # Actual price change %
    was_profitable: bool
    pnl_percent: float
    market_conditions: Dict[str, float] = field(default_factory=dict)
    
    @property
    def prediction_error(self) -> float:
        """Error between predicted and actual movement"""
        return abs(self.signal.predicted_movement - self.actual_movement)
    
    @property
    def direction_correct(self) -> bool:
        """Was the direction prediction correct?"""
        if self.signal.direction == SignalDirection.LONG:
            return self.actual_movement > 0
        elif self.signal.direction == SignalDirection.SHORT:
            return self.actual_movement < 0
        return True  # Neutral is always "correct"


@dataclass
class SignalAccuracyMetrics:
    """Comprehensive accuracy metrics for signals"""
    total_signals: int
    accuracy: float  # Direction accuracy
    precision: float  # True positive rate
    recall: float     # Sensitivity
    f1_score: float
    
    # Financial metrics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    
    # Prediction quality
    avg_prediction_error: float
    correlation_predicted_actual: float
    
    # By confidence level
    accuracy_by_confidence: Dict[str, float]
    
    # By time horizon
    accuracy_by_horizon: Dict[str, float]
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Signal decay
    signal_half_life: float  # Minutes until accuracy drops 50%
    
    def to_dict(self) -> Dict:
        return {
            'total_signals': self.total_signals,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'avg_prediction_error': self.avg_prediction_error,
            'correlation': self.correlation_predicted_actual,
            'signal_half_life': self.signal_half_life,
            'confusion_matrix': {
                'TP': self.true_positives,
                'TN': self.true_negatives,
                'FP': self.false_positives,
                'FN': self.false_negatives
            }
        }


class SignalAccuracyValidator:
    """
    Validates signal accuracy and tracks performance metrics
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.signals: List[TradingSignal] = []
        self.outcomes: List[SignalOutcome] = []
        self.metrics_by_agent: Dict[str, SignalAccuracyMetrics] = {}
        
    def add_signal(self, signal: TradingSignal):
        """Record a new signal"""
        self.signals.append(signal)
        
    def validate_signal(
        self, 
        signal: TradingSignal,
        market_data: pd.DataFrame,
        exit_strategy: str = 'time_based'
    ) -> SignalOutcome:
        """
        Validate a signal against actual market movement
        
        Args:
            signal: The signal to validate
            market_data: Historical data including future prices
            exit_strategy: 'time_based', 'stop_loss', or 'target'
        """
        # Get entry price
        signal_time = signal.timestamp
        if signal_time not in market_data.index:
            # Find nearest timestamp
            idx = market_data.index.get_indexer([signal_time], method='nearest')[0]
            signal_time = market_data.index[idx]
        
        entry_price = market_data.loc[signal_time, 'close']
        
        # Determine exit based on strategy
        if exit_strategy == 'time_based':
            # Exit after specified time horizon
            exit_time = signal_time + timedelta(minutes=signal.time_horizon)
            
            # Find nearest available exit time
            future_data = market_data[market_data.index > signal_time]
            if len(future_data) == 0:
                return None
                
            time_diffs = abs((future_data.index - exit_time).total_seconds())
            exit_idx = time_diffs.argmin()
            exit_time = future_data.index[exit_idx]
            exit_price = future_data.iloc[exit_idx]['close']
            
        elif exit_strategy == 'stop_loss':
            # Exit on stop loss or target
            stop_loss = 0.02  # 2%
            take_profit = abs(signal.predicted_movement) * 1.5
            
            future_data = market_data[market_data.index > signal_time]
            for idx, row in future_data.iterrows():
                price_change = (row['close'] - entry_price) / entry_price
                
                if signal.direction == SignalDirection.LONG:
                    if price_change <= -stop_loss or price_change >= take_profit:
                        exit_time = idx
                        exit_price = row['close']
                        break
                elif signal.direction == SignalDirection.SHORT:
                    if price_change >= stop_loss or price_change <= -take_profit:
                        exit_time = idx
                        exit_price = row['close']
                        break
            else:
                # Use time-based exit as fallback
                exit_time = signal_time + timedelta(minutes=signal.time_horizon)
                exit_idx = future_data.index.get_indexer([exit_time], method='nearest')[0]
                exit_time = future_data.index[exit_idx]
                exit_price = future_data.iloc[exit_idx]['close']
        
        # Calculate actual movement
        actual_movement = (exit_price - entry_price) / entry_price
        
        # Determine if profitable based on direction
        if signal.direction == SignalDirection.LONG:
            pnl_percent = actual_movement
            was_profitable = actual_movement > 0
        elif signal.direction == SignalDirection.SHORT:
            pnl_percent = -actual_movement
            was_profitable = actual_movement < 0
        else:  # NEUTRAL
            pnl_percent = 0
            was_profitable = abs(actual_movement) < 0.005  # Within 0.5%
        
        # Extract market conditions at signal time
        market_conditions = self._extract_market_conditions(market_data, signal_time)
        
        outcome = SignalOutcome(
            signal=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_time=exit_time,
            actual_movement=actual_movement,
            was_profitable=was_profitable,
            pnl_percent=pnl_percent,
            market_conditions=market_conditions
        )
        
        self.outcomes.append(outcome)
        return outcome
    
    def calculate_metrics(
        self, 
        agent_filter: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> SignalAccuracyMetrics:
        """
        Calculate comprehensive accuracy metrics
        
        Args:
            agent_filter: Only include signals from specific agent
            min_confidence: Minimum confidence threshold
        """
        # Filter outcomes
        filtered_outcomes = self.outcomes
        if agent_filter:
            filtered_outcomes = [
                o for o in filtered_outcomes 
                if o.signal.agent_source == agent_filter
            ]
        if min_confidence > 0:
            filtered_outcomes = [
                o for o in filtered_outcomes 
                if o.signal.confidence >= min_confidence
            ]
        
        if not filtered_outcomes:
            return None
        
        # Direction accuracy
        direction_correct = [o.direction_correct for o in filtered_outcomes]
        accuracy = sum(direction_correct) / len(direction_correct)
        
        # Create binary labels for classification metrics
        y_true = []
        y_pred = []
        y_scores = []
        
        for outcome in filtered_outcomes:
            if outcome.signal.direction == SignalDirection.LONG:
                y_true.append(1 if outcome.actual_movement > 0 else 0)
                y_pred.append(1)
                y_scores.append(outcome.signal.confidence)
            elif outcome.signal.direction == SignalDirection.SHORT:
                y_true.append(1 if outcome.actual_movement < 0 else 0)
                y_pred.append(1)
                y_scores.append(outcome.signal.confidence)
        
        # Classification metrics
        if y_true and y_pred:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, len(y_true))
        else:
            precision = recall = f1 = 0
            tn = fp = fn = tp = 0
        
        # Financial metrics
        profitable = [o for o in filtered_outcomes if o.was_profitable]
        losing = [o for o in filtered_outcomes if not o.was_profitable]
        
        win_rate = len(profitable) / len(filtered_outcomes) if filtered_outcomes else 0
        avg_win = np.mean([o.pnl_percent for o in profitable]) if profitable else 0
        avg_loss = np.mean([abs(o.pnl_percent) for o in losing]) if losing else 0
        
        # Profit factor
        gross_profit = sum(o.pnl_percent for o in profitable) if profitable else 0
        gross_loss = sum(abs(o.pnl_percent) for o in losing) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = [o.pnl_percent for o in filtered_outcomes]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Prediction error
        prediction_errors = [o.prediction_error for o in filtered_outcomes]
        avg_prediction_error = np.mean(prediction_errors)
        
        # Correlation between predicted and actual
        predicted = [o.signal.predicted_movement for o in filtered_outcomes]
        actual = [o.actual_movement for o in filtered_outcomes]
        correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0
        
        # Accuracy by confidence level
        confidence_buckets = {
            'low': (0.0, 0.5),
            'medium': (0.5, 0.7),
            'high': (0.7, 1.0)
        }
        accuracy_by_confidence = {}
        
        for bucket_name, (min_conf, max_conf) in confidence_buckets.items():
            bucket_outcomes = [
                o for o in filtered_outcomes 
                if min_conf <= o.signal.confidence < max_conf
            ]
            if bucket_outcomes:
                bucket_correct = sum(o.direction_correct for o in bucket_outcomes)
                accuracy_by_confidence[bucket_name] = bucket_correct / len(bucket_outcomes)
            else:
                accuracy_by_confidence[bucket_name] = 0
        
        # Signal decay analysis
        signal_half_life = self._calculate_signal_decay(filtered_outcomes)
        
        # Accuracy by time horizon
        horizon_buckets = {
            'short': (0, 60),      # < 1 hour
            'medium': (60, 240),   # 1-4 hours
            'long': (240, 1440)    # 4-24 hours
        }
        accuracy_by_horizon = {}
        
        for bucket_name, (min_horizon, max_horizon) in horizon_buckets.items():
            bucket_outcomes = [
                o for o in filtered_outcomes 
                if min_horizon <= o.signal.time_horizon < max_horizon
            ]
            if bucket_outcomes:
                bucket_correct = sum(o.direction_correct for o in bucket_outcomes)
                accuracy_by_horizon[bucket_name] = bucket_correct / len(bucket_outcomes)
            else:
                accuracy_by_horizon[bucket_name] = 0
        
        return SignalAccuracyMetrics(
            total_signals=len(filtered_outcomes),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            avg_prediction_error=avg_prediction_error,
            correlation_predicted_actual=correlation,
            accuracy_by_confidence=accuracy_by_confidence,
            accuracy_by_horizon=accuracy_by_horizon,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            signal_half_life=signal_half_life
        )
    
    def _calculate_signal_decay(self, outcomes: List[SignalOutcome]) -> float:
        """Calculate how quickly signal accuracy degrades over time"""
        if len(outcomes) < 10:
            return float('inf')
        
        # Group by actual time elapsed
        time_buckets = defaultdict(list)
        for outcome in outcomes:
            elapsed_minutes = (outcome.exit_time - outcome.signal.timestamp).total_seconds() / 60
            bucket = int(elapsed_minutes / 30) * 30  # 30-minute buckets
            time_buckets[bucket].append(outcome.direction_correct)
        
        # Calculate accuracy by time bucket
        accuracies = []
        for bucket in sorted(time_buckets.keys()):
            if len(time_buckets[bucket]) >= 5:  # Need minimum samples
                accuracy = sum(time_buckets[bucket]) / len(time_buckets[bucket])
                accuracies.append((bucket, accuracy))
        
        if len(accuracies) < 2:
            return float('inf')
        
        # Find when accuracy drops to 50% of initial
        initial_accuracy = accuracies[0][1]
        target_accuracy = initial_accuracy * 0.5
        
        for minutes, accuracy in accuracies:
            if accuracy <= target_accuracy:
                return minutes
        
        return float('inf')
    
    def _extract_market_conditions(
        self, 
        market_data: pd.DataFrame, 
        timestamp: datetime
    ) -> Dict[str, float]:
        """Extract market conditions at signal time"""
        try:
            # Get recent data
            recent_data = market_data[market_data.index <= timestamp].tail(20)
            
            if len(recent_data) < 2:
                return {}
            
            # Calculate various market metrics
            returns = recent_data['close'].pct_change()
            
            conditions = {
                'volatility': returns.std() * np.sqrt(252),
                'trend': (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0],
                'volume_ratio': recent_data['volume'].iloc[-1] / recent_data['volume'].mean(),
                'rsi': self._calculate_rsi(recent_data['close']),
                'price_position': (recent_data['close'].iloc[-1] - recent_data['low'].min()) / (recent_data['high'].max() - recent_data['low'].min())
            }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error extracting market conditions: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def cross_validate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        k_folds: int = 5
    ) -> Dict[str, SignalAccuracyMetrics]:
        """
        Perform time-series cross-validation
        
        Args:
            market_data: Dictionary of symbol -> price data
            k_folds: Number of folds
            
        Returns:
            Metrics for each fold
        """
        # Sort signals by timestamp
        sorted_signals = sorted(self.signals, key=lambda s: s.timestamp)
        
        if len(sorted_signals) < k_folds * 10:
            logger.warning("Not enough signals for meaningful cross-validation")
            return {}
        
        # Create time-based folds
        fold_size = len(sorted_signals) // k_folds
        fold_metrics = {}
        
        for fold in range(k_folds):
            # Define train and test sets
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(sorted_signals)
            
            test_signals = sorted_signals[test_start:test_end]
            
            # Validate each test signal
            fold_outcomes = []
            for signal in test_signals:
                if signal.symbol in market_data:
                    outcome = self.validate_signal(
                        signal, 
                        market_data[signal.symbol]
                    )
                    if outcome:
                        fold_outcomes.append(outcome)
            
            # Calculate metrics for this fold
            if fold_outcomes:
                # Temporarily replace outcomes
                original_outcomes = self.outcomes
                self.outcomes = fold_outcomes
                
                metrics = self.calculate_metrics()
                fold_metrics[f'fold_{fold+1}'] = metrics
                
                # Restore original outcomes
                self.outcomes = original_outcomes
        
        return fold_metrics
    
    def get_improvement_recommendations(self) -> List[str]:
        """Get recommendations for improving signal quality"""
        recommendations = []
        
        # Analyze overall metrics
        overall_metrics = self.calculate_metrics()
        
        if not overall_metrics:
            return ["Not enough data for recommendations"]
        
        # Check accuracy
        if overall_metrics.accuracy < 0.55:
            recommendations.append(
                "Low direction accuracy - consider reviewing feature engineering "
                "or model selection"
            )
        
        # Check win rate vs average win/loss
        if overall_metrics.win_rate < 0.4 and overall_metrics.avg_win < overall_metrics.avg_loss * 2:
            recommendations.append(
                "Poor risk/reward ratio - need higher win rate or better "
                "profit targets"
            )
        
        # Check prediction error
        if overall_metrics.avg_prediction_error > 0.05:  # 5% error
            recommendations.append(
                "High prediction error - models may need recalibration for "
                "magnitude estimation"
            )
        
        # Check signal decay
        if overall_metrics.signal_half_life < 60:  # Less than 1 hour
            recommendations.append(
                f"Rapid signal decay ({overall_metrics.signal_half_life:.0f} min) - "
                "consider shorter time horizons or faster execution"
            )
        
        # Check confidence calibration
        if overall_metrics.accuracy_by_confidence.get('high', 0) < overall_metrics.accuracy_by_confidence.get('low', 0):
            recommendations.append(
                "Confidence scores not well calibrated - high confidence signals "
                "should be more accurate"
            )
        
        # Agent-specific analysis
        for agent, metrics in self.metrics_by_agent.items():
            if metrics.total_signals > 10:
                if metrics.accuracy < 0.5:
                    recommendations.append(
                        f"Agent '{agent}' has poor accuracy ({metrics.accuracy:.1%}) - "
                        "consider retraining or removing"
                    )
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        overall_metrics = self.calculate_metrics()
        
        report = {
            'summary': {
                'total_signals': len(self.signals),
                'total_validated': len(self.outcomes),
                'overall_accuracy': overall_metrics.accuracy if overall_metrics else 0,
                'overall_profit_factor': overall_metrics.profit_factor if overall_metrics else 0
            },
            'metrics': overall_metrics.to_dict() if overall_metrics else {},
            'by_agent': {},
            'recommendations': self.get_improvement_recommendations()
        }
        
        # Calculate metrics by agent
        agents = set(s.agent_source for s in self.signals)
        for agent in agents:
            agent_metrics = self.calculate_metrics(agent_filter=agent)
            if agent_metrics:
                report['by_agent'][agent] = agent_metrics.to_dict()
                self.metrics_by_agent[agent] = agent_metrics
        
        return report


# Example usage
if __name__ == "__main__":
    # Create validator
    validator = SignalAccuracyValidator()
    
    # Add sample signals
    signal = TradingSignal(
        timestamp=datetime.now(),
        symbol="AAPL",
        direction=SignalDirection.LONG,
        confidence=0.85,
        predicted_movement=0.02,  # 2% up
        time_horizon=120,  # 2 hours
        agent_source="RSI_Agent",
        features={'rsi': 30, 'volume_ratio': 1.5}
    )
    
    validator.add_signal(signal)
    
    # Validate against market data (would need actual data)
    # outcome = validator.validate_signal(signal, market_data)
    
    # Generate report
    report = validator.generate_report()
    print(f"Validation Report: {report}") 