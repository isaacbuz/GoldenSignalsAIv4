import random

class DataSourceAgent:
    def run(self, ticker, timeframe="1D"):
        # Simulate price history (e.g., 50 random prices)
        prices = [round(100 + random.gauss(0, 2), 2) for _ in range(50)]
        # Simulate IV history and current IV
        iv_history = [round(random.uniform(0.15, 0.45), 3) for _ in range(30)]
        current_iv = round(random.uniform(0.15, 0.45), 3)
        # Simulate other fields as needed
        return {
            "price": prices,
            "iv_history": iv_history,
            "current_iv": current_iv,
            # Add more fields if your agents need them
        }
