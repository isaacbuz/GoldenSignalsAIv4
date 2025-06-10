import json
import os
from datetime import datetime
from typing import Dict

class AgentMemoryManager:
    def __init__(self, path="agent_memory.json"):
        self.path = path
        self.memory = self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                return json.load(f)
        return {}

    def record(self, agent_name: str, symbol: str, signal: str, outcome: bool):
        log = self.memory.get(agent_name, [])
        log.append({
            "symbol": symbol,
            "signal": signal,
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.memory[agent_name] = log[-300:]
        self.save()

    def get_accuracy(self, agent_name: str) -> float:
        logs = self.memory.get(agent_name, [])
        if not logs: return 0.5
        return round(sum(1 for l in logs if l['outcome']) / len(logs), 3)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=2)
