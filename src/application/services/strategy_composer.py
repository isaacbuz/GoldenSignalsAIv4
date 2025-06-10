from typing import List, Dict
import pandas as pd

class StrategyComposer:
    def __init__(self, strategy_logic: List[Dict]):
        self.logic_blocks = strategy_logic

    def evaluate(self, df: pd.DataFrame) -> List[str]:
        signals = []
        for i in range(1, len(df)):
            decision = self._evaluate_logic(df.iloc[i-1], df.iloc[i])
            signals.append(decision)
        return ["hold"] + signals

    def _evaluate_logic(self, prev_row, row) -> str:
        for block in self.logic_blocks:
            if block["type"] == "ema_cross":
                ema_short = row.get("ema_9")
                ema_long = row.get("ema_21")
                if ema_short and ema_long:
                    if ema_short > ema_long:
                        return "buy"
                    elif ema_short < ema_long:
                        return "sell"
        return "hold"
