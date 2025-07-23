
"""Mock metrics module for testing"""
from typing import Any


class MetricsCollector:
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        
    def increment(self, metric: str) -> None:
        if metric not in self.metrics:
            self.metrics[metric] = 0
        self.metrics[metric] += 1
        
    def record(self, metric: str, value: float) -> None:
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)
