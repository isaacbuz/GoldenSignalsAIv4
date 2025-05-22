import pytest
import pandas as pd
from agents.correlation_agent import CorrelationAgent

@pytest.fixture
def agent():
    return CorrelationAgent()

def test_compute_correlation_valid(agent):
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [4, 3, 2, 1],
        'C': [1, 2, 1, 2]
    })
    result = agent.compute_correlation(data)
    assert 'correlation_matrix' in result
    assert isinstance(result['correlation_matrix'], dict)

def test_compute_correlation_invalid(agent):
    data = pd.DataFrame({'A': [1, 2, 3, 4]})
    result = agent.compute_correlation(data)
    assert 'error' in result

def test_compute_rolling_correlation_valid(agent):
    s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s2 = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    result = agent.compute_rolling_correlation(s1, s2, window=5)
    assert 'rolling_correlation' in result
    assert isinstance(result['rolling_correlation'], list)

def test_compute_rolling_correlation_invalid(agent):
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([2, 3, 4])
    result = agent.compute_rolling_correlation(s1, s2, window=5)
    assert 'error' in result
