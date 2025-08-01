from typing import Dict, List

import pandas as pd


class PerformanceTracker:
    def __init__(self):
        self.simulated_log = []
        self.live_log = []

    def log_signal(self, symbol: str, signal: str, price: float, mode: str = "live"):
        entry = {
            "symbol": symbol,
            "signal": signal,
            "price": price,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        if mode == "live":
            self.live_log.append(entry)
        else:
            self.simulated_log.append(entry)

    def get_log(self, mode: str = "live") -> List[Dict]:
        return self.live_log if mode == "live" else self.simulated_log

    def compare_performance(self) -> pd.DataFrame:
        live_df = pd.DataFrame(self.live_log)
        sim_df = pd.DataFrame(self.simulated_log)
        if live_df.empty or sim_df.empty:
            return pd.DataFrame()

        # Align signals by timestamp
        live_df["timestamp"] = pd.to_datetime(live_df["timestamp"])
        sim_df["timestamp"] = pd.to_datetime(sim_df["timestamp"])

        merged = pd.merge_asof(
            sim_df.sort_values("timestamp"),
            live_df.sort_values("timestamp"),
            on="timestamp",
            by="symbol",
            suffixes=("_sim", "_live"),
        )
        return merged
