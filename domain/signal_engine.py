"""
Signal Engine: Combines outputs from predictive, sentiment, and risk agents
to produce a unified trade signal.
"""
from typing import Dict, List
import numpy as np

from infrastructure.data_fetcher import MarketDataFetcher
from domain.models.factory import ModelFactory

class SignalEngine:
    def __init__(self, factory: ModelFactory, data_fetcher: MarketDataFetcher):
        self.factory = factory
        self.data_fetcher = data_fetcher

    def generate_trade_signal(self, symbol: str) -> Dict:
        # Step 1: Fetch market data
        market_data = self.data_fetcher.fetch_stock_data(symbol)

        if market_data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        base_signal = {
            "symbol": symbol,
            "raw_data": market_data.tail(5).to_dict(),  # last 5 points as context
        }

        # Step 2: Run predictive agents
        predictive_agents = self.factory.get_predictive_agents()
        predictive_outputs = [agent.process_signal(base_signal.copy()) for agent in predictive_agents]

        # Step 3: Run sentiment agents
        sentiment_agents = self.factory.get_sentiment_agents()
        sentiment_outputs = [agent.process_signal(base_signal.copy()) for agent in sentiment_agents]

        # Step 4: Run risk agents
        risk_agents = self.factory.get_risk_agents()
        risk_outputs = [agent.process_signal(base_signal.copy()) for agent in risk_agents]

        # Step 5: Synthesize results
        call_score = sum(sig.get("call_score", 0) for sig in predictive_outputs + sentiment_outputs) / (len(predictive_outputs) + len(sentiment_outputs))
        put_score = sum(sig.get("put_score", 0) for sig in predictive_outputs + sentiment_outputs) / (len(predictive_outputs) + len(sentiment_outputs))

        action = "CALL" if call_score > put_score else "PUT"
        confidence = round(max(call_score, put_score), 2)

        entry_price = market_data["Close"].iloc[-1]
        exit_price = round(entry_price * (1.03 if action == "CALL" else 0.97), 2)

        risk_notes = [r.get("notes", "") for r in risk_outputs]

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "entry_price": round(entry_price, 2),
            "exit_price": exit_price,
            "reasoning": {
                "predictive": predictive_outputs,
                "sentiment": sentiment_outputs,
                "risk": risk_notes,
            }
        }
