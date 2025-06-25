"""
Tests for the machine learning enhanced signal agent.
"""
import pytest
import numpy as np
import pandas as pd
import os
from agents.research.ml.enhanced_signal_agent import EnhancedSignalAgent

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_points = 100
    
    # Generate price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volume data
    volumes = np.random.lognormal(7, 1, n_points)
    
    return {
        "close": prices.tolist(),
        "open": (prices * 0.99).tolist(),
        "high": (prices * 1.02).tolist(),
        "low": (prices * 0.98).tolist(),
        "volume": volumes.tolist()
    }

@pytest.fixture
def trained_agent(sample_market_data, tmpdir):
    """Create and train an agent for testing."""
    agent = EnhancedSignalAgent()
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_market_data)
    
    # Generate synthetic labels (1: buy, 0: hold, -1: sell)
    returns = df["close"].pct_change()
    labels = pd.Series(np.where(returns > 0.02, 1, np.where(returns < -0.02, -1, 0)))
    
    # Train the agent
    agent.train(df, labels)
    
    # Save the model
    model_path = os.path.join(str(tmpdir), "test_model.joblib")
    agent.save_model(model_path)
    
    return agent, model_path

def test_agent_initialization():
    """Test EnhancedSignalAgent initialization."""
    agent = EnhancedSignalAgent(
        name="TestML",
        lookback_period=20,
        feature_windows=[5, 10, 20]
    )
    
    assert agent.name == "TestML"
    assert agent.agent_type == "ml"
    assert agent.lookback_period == 20
    assert agent.feature_windows == [5, 10, 20]
    assert agent.model is None

def test_feature_calculation(sample_market_data):
    """Test feature calculation."""
    agent = EnhancedSignalAgent()
    df = pd.DataFrame(sample_market_data)
    
    features = agent.calculate_features(df)
    
    # Check feature existence
    for window in agent.feature_windows:
        assert f'return_{window}' in features.columns
        assert f'volatility_{window}' in features.columns
        assert f'momentum_{window}' in features.columns
        assert f'ma_{window}' in features.columns
        assert f'ma_ratio_{window}' in features.columns
    
    assert 'volume_ma' in features.columns
    assert 'volume_ratio' in features.columns
    assert 'high_low_ratio' in features.columns
    assert 'close_open_ratio' in features.columns
    assert 'rsi' in features.columns
    assert 'macd' in features.columns
    assert 'macd_signal' in features.columns
    
    # Check for NaN values
    assert not features.isna().any().any()

def test_model_training(sample_market_data):
    """Test model training process."""
    agent = EnhancedSignalAgent()
    df = pd.DataFrame(sample_market_data)
    
    # Create synthetic labels
    labels = pd.Series(np.random.choice([-1, 0, 1], size=len(df)))
    
    # Train model
    agent.train(df, labels)
    assert agent.model is not None
    assert hasattr(agent.model, 'predict')
    assert hasattr(agent.model, 'predict_proba')

def test_model_persistence(trained_agent, sample_market_data):
    """Test model saving and loading."""
    agent, model_path = trained_agent
    
    # Create new agent and load model
    new_agent = EnhancedSignalAgent()
    new_agent.load_model(model_path)
    
    # Compare predictions
    data = pd.DataFrame(sample_market_data)
    original_pred = agent.process(sample_market_data)
    loaded_pred = new_agent.process(sample_market_data)
    
    assert original_pred["action"] == loaded_pred["action"]
    assert np.allclose(original_pred["confidence"], loaded_pred["confidence"])

def test_signal_generation(trained_agent):
    """Test signal generation."""
    agent, _ = trained_agent
    
    # Test with valid data
    result = agent.process({
        "close": [100, 101, 102],
        "open": [99, 100, 101],
        "high": [102, 103, 104],
        "low": [98, 99, 100],
        "volume": [1000, 1100, 1200]
    })
    
    assert "action" in result
    assert result["action"] in ["buy", "sell", "hold"]
    assert 0 <= result["confidence"] <= 1
    assert "probabilities" in result["metadata"]
    assert "features" in result["metadata"]

def test_error_handling():
    """Test error handling scenarios."""
    agent = EnhancedSignalAgent()
    
    # Test processing without trained model
    result = agent.process({
        "close": [100, 101],
        "volume": [1000, 1100]
    })
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert "error" in result["metadata"]
    
    # Test invalid data format
    result = agent.process({
        "invalid": [1, 2, 3]
    })
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert "error" in result["metadata"]
    
    # Test model saving without training
    with pytest.raises(ValueError):
        agent.save_model("test.joblib")

def test_feature_importance(trained_agent, sample_market_data):
    """Test feature importance accessibility."""
    agent, _ = trained_agent
    
    # Get feature names
    df = pd.DataFrame(sample_market_data)
    features = agent.calculate_features(df)
    feature_names = features.columns.tolist()
    
    # Get feature importances
    importances = agent.model.feature_importances_
    
    assert len(importances) == len(feature_names)
    assert all(imp >= 0 for imp in importances) 