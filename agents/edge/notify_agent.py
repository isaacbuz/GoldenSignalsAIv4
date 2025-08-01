import json
import os
from datetime import datetime

import requests


class NotifyAgent:
    def __init__(self, prefs_path="user_notification_settings.json"):
        self.prefs = self._load(prefs_path)

    def _load(self, path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def run(self, symbol, signal, context):
        if not self.prefs.get("alerts_enabled", False):
            return False

        if signal["signal"] not in self.prefs.get("alert_on", ["bullish"]):
            return False

        if signal["confidence"] < self.prefs.get("confidence_threshold", 0.7):
            return False

        if "source_agents" in context:
            enabled_agents = self.prefs.get("agents", {})
            if not any(enabled_agents.get(a, True) for a in context["source_agents"]):
                return False

        payload = {
            "symbol": symbol,
            "signal": signal["signal"],
            "confidence": signal["confidence"],
            "timestamp": signal.get("timestamp") or datetime.utcnow().isoformat()
        }

        try:
            requests.post("http://localhost:4000/emit", json=payload)
            return True
        except Exception as e:
            print("Notification failed:", e)
            return False
