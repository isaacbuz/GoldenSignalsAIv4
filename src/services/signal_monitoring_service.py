"""
Signal Monitoring Service for GoldenSignalsAI V2
Tracks signal performance, collects feedback, and improves future signals
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SignalOutcome:
    """Represents the outcome of a trading signal"""
    signal_id: str
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    outcome: str  # 'success', 'failure', 'partial', 'pending'
    profit_loss: Optional[float]
    profit_loss_percent: Optional[float]
    holding_period: Optional[timedelta]
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'action': self.action,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'outcome': self.outcome,
            'profit_loss': self.profit_loss,
            'profit_loss_percent': self.profit_loss_percent,
            'holding_period': str(self.holding_period) if self.holding_period else None,
            'notes': self.notes,
            'metadata': self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for signals"""
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    partial_signals: int = 0
    pending_signals: int = 0
    
    win_rate: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    total_profit_loss: float = 0.0
    best_trade: Optional[SignalOutcome] = None
    worst_trade: Optional[SignalOutcome] = None
    
    avg_holding_period: Optional[timedelta] = None
    
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_action: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_risk_level: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'failed_signals': self.failed_signals,
            'partial_signals': self.partial_signals,
            'pending_signals': self.pending_signals,
            'win_rate': self.win_rate,
            'average_profit': self.average_profit,
            'average_loss': self.average_loss,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_profit_loss': self.total_profit_loss,
            'best_trade': self.best_trade.to_dict() if self.best_trade else None,
            'worst_trade': self.worst_trade.to_dict() if self.worst_trade else None,
            'avg_holding_period': str(self.avg_holding_period) if self.avg_holding_period else None,
            'by_symbol': self.by_symbol,
            'by_action': self.by_action,
            'by_risk_level': self.by_risk_level
        }


class SignalMonitoringService:
    """
    Comprehensive signal monitoring service that:
    - Tracks signal outcomes
    - Collects performance metrics
    - Provides feedback for improvement
    - Stores historical performance data
    """
    
    def __init__(self, db_path: str = "signal_monitoring.db"):
        self.db_path = db_path
        self.active_signals: Dict[str, SignalOutcome] = {}
        self.completed_signals: List[SignalOutcome] = []
        self.performance_cache = None
        self.cache_timestamp = None
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_from_database()
        
    def _init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                outcome TEXT NOT NULL,
                profit_loss REAL,
                profit_loss_percent REAL,
                holding_period_seconds INTEGER,
                notes TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                feedback_value TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (signal_id) REFERENCES signal_outcomes(signal_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _load_from_database(self):
        """Load existing data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load completed signals
        cursor.execute("""
            SELECT * FROM signal_outcomes 
            WHERE outcome != 'pending'
            ORDER BY entry_time DESC
            LIMIT 1000
        """)
        
        rows = cursor.fetchall()
        for row in rows:
            outcome = self._row_to_signal_outcome(row)
            self.completed_signals.append(outcome)
            
        # Load active signals
        cursor.execute("""
            SELECT * FROM signal_outcomes 
            WHERE outcome = 'pending'
        """)
        
        rows = cursor.fetchall()
        for row in rows:
            outcome = self._row_to_signal_outcome(row)
            self.active_signals[outcome.signal_id] = outcome
            
        conn.close()
        
    def _row_to_signal_outcome(self, row) -> SignalOutcome:
        """Convert database row to SignalOutcome"""
        return SignalOutcome(
            signal_id=row[0],
            symbol=row[1],
            action=row[2],
            entry_price=row[3],
            exit_price=row[4],
            entry_time=datetime.fromisoformat(row[5]),
            exit_time=datetime.fromisoformat(row[6]) if row[6] else None,
            outcome=row[7],
            profit_loss=row[8],
            profit_loss_percent=row[9],
            holding_period=timedelta(seconds=row[10]) if row[10] else None,
            notes=row[11] or "",
            metadata=json.loads(row[12]) if row[12] else {}
        )
        
    def track_signal_entry(self, signal_data: Dict[str, Any]):
        """Track when a signal is acted upon"""
        signal_id = signal_data.get('id')
        if not signal_id:
            logger.error("Signal ID is required for tracking")
            return
            
        outcome = SignalOutcome(
            signal_id=signal_id,
            symbol=signal_data.get('symbol'),
            action=signal_data.get('action'),
            entry_price=signal_data.get('entry_price', signal_data.get('price')),
            exit_price=None,
            entry_time=datetime.now(),
            exit_time=None,
            outcome='pending',
            profit_loss=None,
            profit_loss_percent=None,
            holding_period=None,
            metadata={
                'confidence': signal_data.get('confidence'),
                'risk_level': signal_data.get('risk_level'),
                'indicators': signal_data.get('indicators', {}),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit': signal_data.get('take_profit')
            }
        )
        
        # Store in active signals
        self.active_signals[signal_id] = outcome
        
        # Save to database
        self._save_signal_outcome(outcome)
        
        logger.info(f"Tracking signal entry: {signal_id} - {outcome.symbol} {outcome.action} @ {outcome.entry_price}")
        
    def track_signal_exit(self, signal_id: str, exit_price: float, outcome: str = 'success', notes: str = ""):
        """Track when a signal position is closed"""
        if signal_id not in self.active_signals:
            logger.warning(f"Signal {signal_id} not found in active signals")
            return
            
        signal_outcome = self.active_signals[signal_id]
        signal_outcome.exit_price = exit_price
        signal_outcome.exit_time = datetime.now()
        signal_outcome.outcome = outcome
        signal_outcome.notes = notes
        
        # Calculate profit/loss
        if signal_outcome.action == "BUY":
            signal_outcome.profit_loss = exit_price - signal_outcome.entry_price
        else:  # SELL
            signal_outcome.profit_loss = signal_outcome.entry_price - exit_price
            
        signal_outcome.profit_loss_percent = (signal_outcome.profit_loss / signal_outcome.entry_price) * 100
        signal_outcome.holding_period = signal_outcome.exit_time - signal_outcome.entry_time
        
        # Determine outcome if not specified
        if outcome == 'success' and signal_outcome.profit_loss < 0:
            signal_outcome.outcome = 'failure'
        elif outcome == 'failure' and signal_outcome.profit_loss > 0:
            signal_outcome.outcome = 'partial'
            
        # Move to completed signals
        self.completed_signals.append(signal_outcome)
        del self.active_signals[signal_id]
        
        # Update database
        self._update_signal_outcome(signal_outcome)
        
        # Invalidate cache
        self.performance_cache = None
        
        logger.info(f"Signal exit tracked: {signal_id} - P/L: ${signal_outcome.profit_loss:.2f} ({signal_outcome.profit_loss_percent:.2f}%)")
        
    def submit_feedback(self, signal_id: str, feedback_type: str, feedback_value: Any):
        """Submit feedback for a signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signal_feedback (signal_id, feedback_type, feedback_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (signal_id, feedback_type, json.dumps(feedback_value), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback submitted for signal {signal_id}: {feedback_type}")
        
    def get_performance_metrics(self, 
                              timeframe: Optional[timedelta] = None,
                              symbol: Optional[str] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Check cache
        if (self.performance_cache and 
            self.cache_timestamp and 
            (datetime.now() - self.cache_timestamp).seconds < self.cache_ttl and
            not timeframe and not symbol):
            return self.performance_cache
            
        # Filter signals based on criteria
        signals = self.completed_signals
        
        if timeframe:
            cutoff_time = datetime.now() - timeframe
            signals = [s for s in signals if s.entry_time >= cutoff_time]
            
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
            
        # Calculate metrics
        metrics = PerformanceMetrics()
        
        if not signals:
            return metrics
            
        # Basic counts
        metrics.total_signals = len(signals) + len(self.active_signals)
        metrics.successful_signals = sum(1 for s in signals if s.outcome == 'success')
        metrics.failed_signals = sum(1 for s in signals if s.outcome == 'failure')
        metrics.partial_signals = sum(1 for s in signals if s.outcome == 'partial')
        metrics.pending_signals = len(self.active_signals)
        
        # Win rate
        completed_count = metrics.successful_signals + metrics.failed_signals + metrics.partial_signals
        if completed_count > 0:
            metrics.win_rate = (metrics.successful_signals + metrics.partial_signals * 0.5) / completed_count
            
        # Profit/Loss metrics
        profits = [s.profit_loss for s in signals if s.profit_loss and s.profit_loss > 0]
        losses = [abs(s.profit_loss) for s in signals if s.profit_loss and s.profit_loss < 0]
        
        if profits:
            metrics.average_profit = np.mean(profits)
            
        if losses:
            metrics.average_loss = np.mean(losses)
            
        if profits and losses:
            metrics.profit_factor = sum(profits) / sum(losses)
            
        # Total P/L
        all_pnl = [s.profit_loss for s in signals if s.profit_loss is not None]
        if all_pnl:
            metrics.total_profit_loss = sum(all_pnl)
            
            # Sharpe ratio (simplified)
            returns = [s.profit_loss_percent for s in signals if s.profit_loss_percent is not None]
            if len(returns) > 1:
                metrics.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                
        # Max drawdown
        if all_pnl:
            cumulative_pnl = np.cumsum(all_pnl)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = (cumulative_pnl - running_max) / running_max
            metrics.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
        # Best and worst trades
        if signals:
            sorted_by_pnl = sorted([s for s in signals if s.profit_loss is not None], 
                                 key=lambda x: x.profit_loss)
            if sorted_by_pnl:
                metrics.worst_trade = sorted_by_pnl[0]
                metrics.best_trade = sorted_by_pnl[-1]
                
        # Average holding period
        holding_periods = [s.holding_period for s in signals if s.holding_period]
        if holding_periods:
            avg_seconds = np.mean([hp.total_seconds() for hp in holding_periods])
            metrics.avg_holding_period = timedelta(seconds=avg_seconds)
            
        # Breakdown by symbol
        for signal in signals:
            if signal.symbol not in metrics.by_symbol:
                metrics.by_symbol[signal.symbol] = {
                    'count': 0,
                    'win_rate': 0,
                    'total_pnl': 0
                }
                
            metrics.by_symbol[signal.symbol]['count'] += 1
            if signal.profit_loss:
                metrics.by_symbol[signal.symbol]['total_pnl'] += signal.profit_loss
                
        # Calculate win rates by symbol
        for symbol in metrics.by_symbol:
            symbol_signals = [s for s in signals if s.symbol == symbol]
            wins = sum(1 for s in symbol_signals if s.outcome in ['success', 'partial'])
            total = len(symbol_signals)
            if total > 0:
                metrics.by_symbol[symbol]['win_rate'] = wins / total
                
        # Breakdown by action
        for action in ['BUY', 'SELL']:
            action_signals = [s for s in signals if s.action == action]
            if action_signals:
                wins = sum(1 for s in action_signals if s.outcome in ['success', 'partial'])
                metrics.by_action[action] = {
                    'count': len(action_signals),
                    'win_rate': wins / len(action_signals) if len(action_signals) > 0 else 0,
                    'avg_pnl': np.mean([s.profit_loss for s in action_signals if s.profit_loss is not None])
                }
                
        # Cache results
        if not timeframe and not symbol:
            self.performance_cache = metrics
            self.cache_timestamp = datetime.now()
            
        return metrics
        
    def get_signal_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of signal feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get feedback counts
        cursor.execute("""
            SELECT feedback_type, COUNT(*) 
            FROM signal_feedback 
            GROUP BY feedback_type
        """)
        
        feedback_counts = dict(cursor.fetchall())
        
        # Get recent feedback
        cursor.execute("""
            SELECT signal_id, feedback_type, feedback_value, timestamp
            FROM signal_feedback
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        recent_feedback = []
        for row in cursor.fetchall():
            recent_feedback.append({
                'signal_id': row[0],
                'feedback_type': row[1],
                'feedback_value': json.loads(row[2]),
                'timestamp': row[3]
            })
            
        conn.close()
        
        return {
            'feedback_counts': feedback_counts,
            'recent_feedback': recent_feedback,
            'total_feedback': sum(feedback_counts.values())
        }
        
    def generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for improving signal generation"""
        metrics = self.get_performance_metrics()
        recommendations = []
        
        # Win rate recommendations
        if metrics.win_rate < 0.5:
            recommendations.append({
                'type': 'win_rate',
                'severity': 'high',
                'message': f'Win rate is {metrics.win_rate:.1%}. Consider increasing signal quality thresholds.',
                'action': 'increase_confidence_threshold'
            })
            
        # Profit factor recommendations
        if metrics.profit_factor < 1.5:
            recommendations.append({
                'type': 'profit_factor',
                'severity': 'medium',
                'message': f'Profit factor is {metrics.profit_factor:.2f}. Focus on improving risk/reward ratios.',
                'action': 'adjust_stop_loss_take_profit'
            })
            
        # Symbol-specific recommendations
        for symbol, stats in metrics.by_symbol.items():
            if stats['win_rate'] < 0.4:
                recommendations.append({
                    'type': 'symbol_performance',
                    'severity': 'medium',
                    'message': f'{symbol} has low win rate ({stats["win_rate"]:.1%}). Consider excluding or adjusting parameters.',
                    'action': 'review_symbol_parameters',
                    'symbol': symbol
                })
                
        # Action-specific recommendations
        for action, stats in metrics.by_action.items():
            if stats['win_rate'] < 0.45:
                recommendations.append({
                    'type': 'action_performance',
                    'severity': 'low',
                    'message': f'{action} signals underperforming ({stats["win_rate"]:.1%} win rate).',
                    'action': 'review_action_criteria',
                    'signal_action': action
                })
                
        return recommendations
        
    def _save_signal_outcome(self, outcome: SignalOutcome):
        """Save signal outcome to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO signal_outcomes 
            (signal_id, symbol, action, entry_price, exit_price, entry_time, 
             exit_time, outcome, profit_loss, profit_loss_percent, 
             holding_period_seconds, notes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.signal_id,
            outcome.symbol,
            outcome.action,
            outcome.entry_price,
            outcome.exit_price,
            outcome.entry_time.isoformat(),
            outcome.exit_time.isoformat() if outcome.exit_time else None,
            outcome.outcome,
            outcome.profit_loss,
            outcome.profit_loss_percent,
            outcome.holding_period.total_seconds() if outcome.holding_period else None,
            outcome.notes,
            json.dumps(outcome.metadata)
        ))
        
        conn.commit()
        conn.close()
        
    def _update_signal_outcome(self, outcome: SignalOutcome):
        """Update signal outcome in database"""
        self._save_signal_outcome(outcome)
        
    def save_performance_snapshot(self):
        """Save current performance metrics snapshot"""
        metrics = self.get_performance_metrics()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_snapshots (timestamp, metrics)
            VALUES (?, ?)
        """, (datetime.now().isoformat(), json.dumps(metrics.to_dict())))
        
        conn.commit()
        conn.close()
        
        logger.info("Performance snapshot saved") 