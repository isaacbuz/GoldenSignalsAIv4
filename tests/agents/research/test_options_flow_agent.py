import pytest
import pandas as pd
from agents.research.ml.options_flow import OptionsFlowAgent

def test_options_flow_agent():
    data = {
        'options_data': [
            {'iv': 0.25, 'skew': 0.12, 'volume': 800},
            {'iv': 0.30, 'skew': 0.15, 'volume': 1000},
        ]
    }
    agent = OptionsFlowAgent()
    signal = agent.process(data)
    assert signal['action'] in ['buy', 'sell', 'hold']
    assert isinstance(signal, dict)
