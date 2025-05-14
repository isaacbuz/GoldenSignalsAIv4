import os
import sys
import pytest
import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope='session')
def sample_market_data():
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    return {
        'stock_prices': np.random.normal(100, 20, (100, 5)),
        'options_volume': np.random.poisson(500, (100, 3)),
        'sentiment_scores': np.random.uniform(-1, 1, 100)
    }

@pytest.fixture(scope='session')
def ml_model_fixtures():
    """Provide fixtures for machine learning model testing."""
    from domain.models.ai_models import LSTMModel, TransformerModel
    
    lstm_model = LSTMModel(input_dim=5, hidden_dim=32)
    transformer_model = TransformerModel(input_dim=5, hidden_dim=32)
    
    return {
        'lstm': lstm_model,
        'transformer': transformer_model
    }

@pytest.fixture(scope='session')
def torch_device():
    """Determine and return the appropriate torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", 
        "performance: mark test related to performance metrics"
    )
