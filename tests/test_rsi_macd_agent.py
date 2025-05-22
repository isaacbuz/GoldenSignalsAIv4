import pandas as pd
from agents.rsi_macd_agent import RSIMACDAgent

def test_rsi_macd_signal():
    agent = RSIMACDAgent()
    df = pd.DataFrame({"close": [100, 102, 101, 105, 110, 115, 120, 118, 117, 119, 121, 122, 120, 119, 120]})
    signal = agent.compute_signal(df)
    assert signal in ["buy", "sell", "hold"]
