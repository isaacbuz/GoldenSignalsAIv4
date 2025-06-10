"""
Tests for AlphaPy modeling functionality.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from AlphaPy.models import (
    AlphaModel,
    EnsembleModel,
    FactorModel,
    TimeSeriesModel
)

@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample data for model testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, len(dates)),
        'feature2': np.random.normal(0, 1, len(dates)),
        'feature3': np.random.normal(0, 1, len(dates))
    }, index=dates)
    
    # Create target (with some relationship to features)
    y = pd.Series(
        0.3 * X['feature1'] + 0.5 * X['feature2'] - 0.2 * X['feature3'] + np.random.normal(0, 0.1, len(dates)),
        index=dates
    )
    
    return X, y

class SimpleAlphaModel(AlphaModel):
    """Simple implementation of AlphaModel for testing."""
    
    def train(self, X, y, validation_data=None):
        self.mean = y.mean()
        
    def predict(self, X):
        return np.full(len(X), self.mean)
        
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = ((predictions - y) ** 2).mean()
        return {'mse': mse}

def test_alpha_model_base():
    """Test base AlphaModel functionality."""
    config = {'param1': 'value1'}
    model = SimpleAlphaModel(config)
    
    assert model.config == config
    assert model.model is None
    assert model.features == []

def test_ensemble_model(sample_data):
    """Test EnsembleModel functionality."""
    X, y = sample_data
    
    # Create ensemble
    ensemble = EnsembleModel({'name': 'test_ensemble'})
    
    # Add models with weights
    model1 = SimpleAlphaModel({'name': 'model1'})
    model2 = SimpleAlphaModel({'name': 'model2'})
    
    ensemble.add_model(model1, weight=0.6)
    ensemble.add_model(model2, weight=0.4)
    
    # Test weight normalization
    ensemble.normalize_weights()
    assert np.allclose(sum(ensemble.weights), 1.0)
    
    # Verify models were added
    assert len(ensemble.models) == 2
    assert len(ensemble.weights) == 2

def test_factor_model(sample_data):
    """Test FactorModel functionality."""
    X, y = sample_data
    
    model = FactorModel({'name': 'test_factor_model'})
    
    # Add factors
    model.add_factor('factor1', X['feature1'])
    model.add_factor('factor2', X['feature2'])
    
    # Verify factors were added
    assert len(model.factors) == 2
    assert 'factor1' in model.exposures.columns
    assert 'factor2' in model.exposures.columns

def test_time_series_model(sample_data):
    """Test TimeSeriesModel functionality."""
    X, y = sample_data
    
    config = {
        'lookback': 5,
        'horizon': 1
    }
    model = TimeSeriesModel(config)
    
    # Test sequence preparation
    sequences, targets = model.prepare_sequences(X)
    
    # Verify sequence dimensions
    assert len(sequences) == len(X) - model.lookback - model.horizon + 1
    assert sequences.shape[1] == model.lookback
    assert sequences.shape[2] == X.shape[1]  # number of features
    
    # Verify configuration
    assert model.lookback == config['lookback']
    assert model.horizon == config['horizon']

@pytest.mark.parametrize("lookback,horizon", [
    (5, 1),
    (10, 2),
    (20, 5)
])
def test_time_series_model_parameters(sample_data, lookback, horizon):
    """Test TimeSeriesModel with different parameters."""
    X, y = sample_data
    
    model = TimeSeriesModel({
        'lookback': lookback,
        'horizon': horizon
    })
    
    sequences, targets = model.prepare_sequences(X)
    
    # Verify dimensions with different parameters
    assert sequences.shape[1] == lookback
    assert len(sequences) == len(X) - lookback - horizon + 1 