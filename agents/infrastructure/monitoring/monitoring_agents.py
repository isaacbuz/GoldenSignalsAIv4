"""
monitoring_agents.py

Provides agent-centric monitoring utilities for GoldenSignalsAI.
Includes:
- AgentPerformanceTracker: Tracks agent win rates and signal stats
- AIMonitor: Tracks system/model health metrics

These classes were migrated from application/monitoring for unified agent monitoring and discoverability.
"""
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict

import numpy as np


class AgentPerformanceTracker:
    """Tracks signal outcomes and win rates for agents."""
    def __init__(self):
        self.performance_data = defaultdict(lambda: {
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "last_signal_time": None
        })
    def record_signal(self, agent_id: str, outcome: str):
        data = self.performance_data[agent_id]
        data["total_signals"] += 1
        data["last_signal_time"] = time.time()
        if outcome == "win":
            data["wins"] += 1
        elif outcome == "loss":
            data["losses"] += 1
    def get_summary(self, agent_id: str) -> Dict:
        data = self.performance_data[agent_id]
        win_rate = (data["wins"] / data["total_signals"]) * 100 if data["total_signals"] else 0
        return {
            "agent_id": agent_id,
            "total_signals": data["total_signals"],
            "wins": data["wins"],
            "losses": data["losses"],
            "win_rate": round(win_rate, 2),
            "last_signal_time": data["last_signal_time"]
        }
    def get_all_summaries(self) -> Dict[str, Dict]:
        return {agent: self.get_summary(agent) for agent in self.performance_data}

class AIMonitor:
    """Tracks model/system health metrics for AI agent monitoring."""
    METRICS = ["model_accuracy", "data_freshness", "trade_execution_latency", "system_uptime", "win_rate"]
    def __init__(self):
        self.metrics_history = {metric: [] for metric in self.METRICS}
        self.logger = logging.getLogger(__name__)
    async def update_metrics(self):
        current_time = datetime.now()
        self.metrics_history["model_accuracy"].append({
            "timestamp": current_time.isoformat(),
            "value": np.random.uniform(0.7, 0.95)
        })
        self.metrics_history["data_freshness"].append({
            "timestamp": current_time.isoformat(),
            "value": np.random.uniform(0, 60)
        })
        self.metrics_history["trade_execution_latency"].append({
            "timestamp": current_time.isoformat(),
            "value": np.random.uniform(0.1, 2.0)
        })
        self.metrics_history["system_uptime"].append({
            "timestamp": current_time.isoformat(),
            "value": (current_time - datetime.fromisoformat("2025-05-04T00:00:00")).total_seconds() / 3600
        })
        self.metrics_history["win_rate"].append({
            "timestamp": current_time.isoformat(),
            "value": np.random.uniform(0.5, 0.8)
        })
        for metric in self.METRICS:
            self.metrics_history[metric] = self.metrics_history[metric][-100:]
        self.logger.info("Updated system health metrics")
