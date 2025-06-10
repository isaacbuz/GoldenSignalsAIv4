import random
from typing import Dict, Any

class ArbitrageExecutor:
    def __init__(self, slippage_tolerance=0.2, latency_ms=150, min_volume=100, fee_perc=0.0005):
        self.slippage_tolerance = slippage_tolerance  # $ or %
        self.latency_ms = latency_ms
        self.min_volume = min_volume
        self.fee_perc = fee_perc

    def simulate(self, opportunity: Dict[str, Any], orderbook: Dict[str, Any] = None) -> Dict[str, Any]:
        # Simulate slippage, latency, and execution risk
        spread = opportunity.get("spread", 0)
        buy_price = opportunity.get("buy_price", 0)
        sell_price = opportunity.get("sell_price", 0)
        symbol = opportunity.get("symbol", "")
        volume = opportunity.get("volume", self.min_volume)

        # Simulate execution delay
        simulated_latency = random.randint(self.latency_ms, self.latency_ms + 200)

        # Simulate slippage
        slippage = random.uniform(0, self.slippage_tolerance)
        executed_buy = buy_price + slippage
        executed_sell = sell_price - slippage

        # Simulate fill probability
        fill_probability = min(1.0, max(0.2, spread / (slippage + 0.01)))
        filled = random.random() < fill_probability

        # Simulate fee
        fees = self.fee_perc * (executed_buy + executed_sell) * volume
        pnl = (executed_sell - executed_buy) * volume - fees if filled else 0

        status = "Executed" if filled else "Missed"

        return {
            "symbol": symbol,
            "status": status,
            "executed_buy": round(executed_buy, 4),
            "executed_sell": round(executed_sell, 4),
            "volume": volume,
            "fill_probability": round(fill_probability, 3),
            "latency_ms": simulated_latency,
            "fees": round(fees, 4),
            "pnl": round(pnl, 4)
        }
