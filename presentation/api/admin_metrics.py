# admin_metrics.py
# Utilities for system and agent metrics (in-memory time series for demo)
import psutil
import time
from collections import deque
from typing import Dict, Any

# Store recent metrics (last 1 hour, 60 points)
METRIC_HISTORY = deque(maxlen=60)
LAST_METRIC_TIME = 0

# Dummy agent health (should be replaced with real agent monitoring)
AGENT_HEALTH = {
    "AlphaVantageAgent": {"status": "active", "last_heartbeat": time.time(), "latency": 0.2, "error_rate": 0.01},
    "FinnhubAgent": {"status": "active", "last_heartbeat": time.time(), "latency": 0.3, "error_rate": 0.0},
    "PolygonAgent": {"status": "inactive", "last_heartbeat": time.time() - 120, "latency": 0.0, "error_rate": 0.05},
}

# For queue/task monitoring (dummy)
QUEUE_STATUS = {"depth": 3, "workers": 2, "active": 1}

def collect_metrics() -> Dict[str, Any]:
    return {
        "timestamp": int(time.time()),
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().used // (1024 * 1024),
        "uptime": int(time.time() - psutil.boot_time()),
        "activeRequests": 7  # Replace with real count if available
    }

def update_metric_history():
    global LAST_METRIC_TIME
    now = time.time()
    if now - LAST_METRIC_TIME > 60 or len(METRIC_HISTORY) == 0:
        METRIC_HISTORY.append(collect_metrics())
        LAST_METRIC_TIME = now

def get_metric_history():
    update_metric_history()
    return list(METRIC_HISTORY)

def get_agent_health():
    # In a real app, update agent health from live agent pings
    return AGENT_HEALTH

def get_queue_status():
    return QUEUE_STATUS
