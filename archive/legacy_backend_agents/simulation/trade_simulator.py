import pandas as pd
from typing import Dict, List

class TradeSimulator:
    def __init__(self, slippage: float = 0.001, fee_perc: float = 0.001):
        self.slippage = slippage
        self.fee = fee_perc

    def simulate_trades(self, signals: List[Dict], price_data: pd.DataFrame, holding_days: int = 5):
        trades = []
        price_data = price_data.set_index("date")
        for signal in signals:
            date = signal["date"]
            direction = signal["signal"]
            confidence = signal.get("confidence", 0.5)
            symbol = signal.get("symbol", "UNKNOWN")
            if date not in price_data.index:
                continue
            entry_price = price_data.loc[date, "close"]
            entry_price *= (1 + self.slippage + self.fee)
            try:
                exit_date = price_data.index[price_data.index.get_loc(date) + holding_days]
            except:
                continue
            exit_price = price_data.loc[exit_date, "close"]
            exit_price *= (1 - self.slippage - self.fee)
            pct_return = (exit_price - entry_price) / entry_price
            if direction == "bearish":
                pct_return *= -1
            trades.append({
                "symbol": symbol,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(pct_return, 4),
                "confidence": confidence,
                "direction": direction
            })
        return pd.DataFrame(trades)
