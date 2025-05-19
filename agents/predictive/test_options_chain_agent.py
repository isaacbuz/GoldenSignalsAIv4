import pytest
import pandas as pd
from agents.predictive.options_chain import OptionsChainAgent

def test_options_chain_agent():
    data = {
        'options_chain': [
            {'strike': 100, 'open_interest': 1200, 'volume': 800, 'sentiment': 'bullish', 'sentiment_score': 0.7},
            {'strike': 105, 'open_interest': 1500, 'volume': 1000, 'sentiment': 'bearish', 'sentiment_score': -0.6},
        ]
    }
    agent = OptionsChainAgent()
    signal = agent.process(data)
    assert signal['action'] in ['buy', 'sell', 'hold']
    assert isinstance(signal, dict)
