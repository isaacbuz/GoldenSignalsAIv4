"""
Signal management system for handling and processing trading signals.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class Signal:
    def __init__(
        self,
        agent: str,
        signal_type: str,
        action: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.agent = agent
        self.signal_type = signal_type
        self.action = action
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "type": self.signal_type,
            "action": self.action,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class SignalManager:
    def __init__(self):
        self.signals: List[Signal] = []
        self.weight_config = {
            "technical": 0.4,
            "sentiment": 0.3,
            "ml": 0.3
        }

    def add_signal(self, signal: Signal) -> None:
        """Add a new signal to the manager"""
        self.signals.append(signal)

    def clear_signals(self) -> None:
        """Clear all signals"""
        self.signals = []

    def get_recent_signals(self, minutes: int = 5) -> List[Signal]:
        """Get signals from the last N minutes"""
        cutoff = datetime.now().timestamp() - (minutes * 60)
        return [s for s in self.signals if s.timestamp.timestamp() > cutoff]

    def aggregate_signals(self) -> Dict[str, Any]:
        """Aggregate all current signals into a single decision"""
        if not self.signals:
            return {
                "action": "hold",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "contributing_signals": []
            }

        # Calculate weighted votes
        votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        for signal in self.signals:
            weight = self.weight_config.get(signal.signal_type, 0.2)
            votes[signal.action] += signal.confidence * weight

        # Determine final action
        final_action = max(votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(votes.values())
        confidence = votes[final_action] / total_weight if total_weight > 0 else 0.0

        return {
            "action": final_action,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "contributing_signals": [s.to_dict() for s in self.signals[-3:]]  # Last 3 signals
        }

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update signal type weights"""
        self.weight_config.update(new_weights)
