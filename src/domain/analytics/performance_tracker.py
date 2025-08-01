"""
Performance Tracker: Tracks signal performance and accuracy metrics.
"""
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np


class PerformanceTracker:
    def __init__(self, max_signals: int = 1000):
        self.signals = deque(maxlen=max_signals)
        self.performance_metrics = {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_confidence": 0.0,
            "average_validation_score": 0.0,
        }
        self._update_interval = timedelta(minutes=5)
        self._last_update = datetime.now()

    def track_signal(self, signal: Dict) -> None:
        """
        Track a new signal and update performance metrics.

        Args:
            signal: Dictionary containing signal information
        """
        self.signals.append({"signal": signal, "timestamp": datetime.now(), "status": "pending"})

        # Update metrics if enough time has passed
        if datetime.now() - self._last_update >= self._update_interval:
            self._update_metrics()

    def update_signal_status(self, symbol: str, status: str, profit_loss: float = 0.0) -> None:
        """
        Update the status of a signal and its profit/loss.

        Args:
            symbol: Symbol of the signal to update
            status: New status ('success' or 'failed')
            profit_loss: Profit or loss amount
        """
        for signal_data in self.signals:
            if signal_data["signal"]["symbol"] == symbol:
                signal_data["status"] = status
                signal_data["profit_loss"] = profit_loss
                break

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        self._update_metrics()
        return self.performance_metrics

    def get_signal_accuracy(self) -> float:
        """Get current signal accuracy."""
        self._update_metrics()
        return self.performance_metrics["win_rate"]

    def _update_metrics(self) -> None:
        """Update performance metrics based on tracked signals."""
        if not self.signals:
            return

        total_signals = len(self.signals)
        successful_signals = sum(1 for s in self.signals if s["status"] == "success")
        failed_signals = sum(1 for s in self.signals if s["status"] == "failed")

        total_profit = sum(s["profit_loss"] for s in self.signals if s["profit_loss"] > 0)
        total_loss = abs(sum(s["profit_loss"] for s in self.signals if s["profit_loss"] < 0))

        win_rate = successful_signals / total_signals if total_signals > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        avg_confidence = np.mean([s["signal"]["confidence"] for s in self.signals])
        avg_validation = np.mean([s["signal"]["validation_score"] for s in self.signals])

        self.performance_metrics.update(
            {
                "total_signals": total_signals,
                "successful_signals": successful_signals,
                "failed_signals": failed_signals,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "average_confidence": avg_confidence,
                "average_validation_score": avg_validation,
            }
        )

        self._last_update = datetime.now()

    def get_recent_signals(self, n: int = 10) -> List[Dict]:
        """Get the n most recent signals."""
        return list(self.signals)[-n:]

    def get_signal_history(self, symbol: str) -> List[Dict]:
        """Get signal history for a specific symbol."""
        return [s for s in self.signals if s["signal"]["symbol"] == symbol]

    def get_performance_by_confidence(self) -> Dict:
        """Get performance metrics grouped by confidence levels."""
        confidence_ranges = {"high": (0.8, 1.0), "medium": (0.6, 0.8), "low": (0.0, 0.6)}

        performance = {}
        for level, (min_conf, max_conf) in confidence_ranges.items():
            signals = [s for s in self.signals if min_conf <= s["signal"]["confidence"] < max_conf]

            if signals:
                successful = sum(1 for s in signals if s["status"] == "success")
                total = len(signals)
                performance[level] = {
                    "total_signals": total,
                    "successful_signals": successful,
                    "win_rate": successful / total if total > 0 else 0.0,
                }
            else:
                performance[level] = {"total_signals": 0, "successful_signals": 0, "win_rate": 0.0}

        return performance
