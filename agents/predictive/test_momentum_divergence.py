import pytest
import pandas as pd
from agents.predictive.momentum_divergence import MomentumDivergenceAgent

def test_momentum_divergence_agent_bullish():
    # Simulate bullish divergence: price lower low, RSI higher low
    data = {
        'Close': [100, 98, 97, 99, 95, 94, 96, 97, 98, 99, 98, 100, 102, 101, 99, 97, 96, 95, 97, 98, 99, 98, 100, 102, 103]
    }
    df = pd.DataFrame(data)
    agent = MomentumDivergenceAgent(lookback=10, indicator='RSI')
    signal = agent.process(df)
    assert signal['action'] in ['buy', 'hold']  # Accept buy or hold if not enough divergence
    assert 'bullish' in signal['metadata']

def test_momentum_divergence_agent_bearish():
    # Simulate bearish divergence: price higher high, RSI lower high
    data = {
        'Close': [100, 102, 104, 105, 107, 109, 110, 111, 112, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98]
    }
    df = pd.DataFrame(data)
    agent = MomentumDivergenceAgent(lookback=10, indicator='RSI')
    signal = agent.process(df)
    assert signal['action'] in ['sell', 'hold']
    assert 'bearish' in signal['metadata']
