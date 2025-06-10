import pandas as pd
from datetime import datetime
from typing import Dict, List
from src.application.services.signal_engine import SignalEngine

class MultiTimeframeSignalEngine:
    def __init__(self):
        self.engines = {
            '1m': SignalEngine(interval='1m'),
            '5m': SignalEngine(interval='5m'),
            '1h': SignalEngine(interval='1h'),
            '1d': SignalEngine(interval='1d')
        }

    def generate_signals(self, symbol: str, data_by_interval: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        results = {}
        for interval, engine in self.engines.items():
            if interval in data_by_interval:
                df = data_by_interval[interval]
                signal_result = engine.compute_signal(df)
                results[interval] = {
                    "signal": signal_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
        return results
