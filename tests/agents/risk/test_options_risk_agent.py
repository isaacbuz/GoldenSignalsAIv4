import pytest
from agents.risk.options_risk import OptionsRiskAgent
import pandas as pd

def test_options_risk_agent():
    data = {
        'options_data': pd.DataFrame({
            'delta': [0.5, 0.8, 0.6],
            'gamma': [0.05, 0.12, 0.09],
            'theta': [-0.03, -0.07, -0.04],
        })
    }
    agent = OptionsRiskAgent()
    signal = agent.process(data)
    assert signal['action'] in ['buy', 'sell', 'hold']
    assert isinstance(signal, dict)
