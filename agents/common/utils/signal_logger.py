"""
Signal logging utility for tracking and analyzing trading signals.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

class SignalLogger:
    """Logger for trading signals with analysis capabilities."""

    def __init__(self, log_dir: str = "logs/signals"):
        """
        Initialize signal logger.

        Args:
            log_dir: Directory to store signal logs
        """
        self.log_dir = log_dir
        self._ensure_log_dir()
        self.current_log = None
        self.signals = []

    def _ensure_log_dir(self):
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)

    def start_new_log(self, name: str):
        """Start a new signal log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log = os.path.join(self.log_dir, f"{name}_{timestamp}.json")
        self.signals = []

    def log_signal(self, signal: Dict[str, Any]):
        """Log a trading signal."""
        if not self.current_log:
            raise ValueError("No active log file. Call start_new_log() first.")

        signal_entry = {
            "timestamp": datetime.now().isoformat(),
            "signal": signal
        }
        self.signals.append(signal_entry)

        # Write to file
        with open(self.current_log, "w") as f:
            json.dump({"signals": self.signals}, f, indent=2)

    def analyze_signals(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze logged signals within date range.

        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            Dict containing signal analysis
        """
        if not self.signals:
            return {"error": "No signals logged"}

        # Convert signals to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": s["timestamp"],
                "action": s["signal"].get("action"),
                "confidence": s["signal"].get("confidence", 0.0)
            }
            for s in self.signals
        ])

        # Apply date filters
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        # Calculate metrics
        total_signals = len(df)
        signal_counts = df["action"].value_counts().to_dict()
        avg_confidence = df["confidence"].mean()

        # Calculate signal transition matrix
        transitions = pd.crosstab(
            df["action"].shift(),
            df["action"],
            normalize="index"
        ).round(2).to_dict()

        return {
            "total_signals": total_signals,
            "signal_distribution": signal_counts,
            "average_confidence": avg_confidence,
            "signal_transitions": transitions,
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat()
            }
        }

    def load_log(self, log_file: str) -> bool:
        """
        Load signals from an existing log file.

        Args:
            log_file: Path to log file

        Returns:
            bool: True if successful
        """
        try:
            with open(os.path.join(self.log_dir, log_file), "r") as f:
                data = json.load(f)
            self.signals = data.get("signals", [])
            self.current_log = os.path.join(self.log_dir, log_file)
            return True
        except Exception as e:
            logger.error(f"Failed to load signal log: {e}")
            return False

    def get_recent_signals(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent n signals.

        Args:
            n: Number of signals to return

        Returns:
            List of recent signals
        """
        return self.signals[-n:] if self.signals else []
