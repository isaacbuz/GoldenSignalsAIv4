import pytest
import pandas as pd
from agents.research.ml.reversion import ReversionAgent

def test_reversion_agent_basic():
    data = {
        'stock_data': pd.DataFrame({'Close': [100, 102, 101, 99, 98, 97, 98, 99, 100, 101, 102, 103, 104, 105]})
    }
    agent = ReversionAgent()
    signal = agent.process(data)
    assert signal['action'] in ['buy', 'sell', 'hold']
    assert isinstance(signal, dict)
