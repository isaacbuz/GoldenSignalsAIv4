import time
from collections import defaultdict
from typing import Dict

class AgentPerformanceTracker:
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
