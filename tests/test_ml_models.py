import pytest
import numpy as np
import torch

def test_lstm_model_initialization(ml_model_fixtures):
    """Test LSTM model initialization."""
    lstm_model = ml_model_fixtures['lstm']
    assert lstm_model is not None
    assert hasattr(lstm_model, 'lstm')
    assert hasattr(lstm_model, 'fc')

def test_transformer_model_initialization(ml_model_fixtures):
    """Test Transformer model initialization."""
    transformer_model = ml_model_fixtures['transformer']
    assert transformer_model is not None
    assert hasattr(transformer_model, 'embedding')
    assert hasattr(transformer_model, 'transformer_encoder')
    assert hasattr(transformer_model, 'fc')

@pytest.mark.parametrize("model_type", ['lstm', 'transformer'])
def test_model_forward_pass(ml_model_fixtures, sample_market_data, model_type):
    """Test forward pass for different model types."""
    model = ml_model_fixtures[model_type]
    stock_prices = sample_market_data['stock_prices']

    # Prepare input tensor
    input_tensor = torch.FloatTensor(stock_prices[:10]).unsqueeze(0)

    # Perform forward pass
    output = model(input_tensor)

    assert output is not None
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # Batch size
    assert output.shape[1] == 1  # Single output value

@pytest.mark.performance
def test_model_performance_characteristics(ml_model_fixtures, sample_market_data):
    """Evaluate basic performance characteristics of models."""
    for model_name, model in ml_model_fixtures.items():
        stock_prices = sample_market_data['stock_prices']
        input_tensor = torch.FloatTensor(stock_prices[:50]).unsqueeze(0)

        # Measure inference time
        import time
        start_time = time.time()
        _ = model(input_tensor)
        inference_time = time.time() - start_time

        assert inference_time < 0.1, f"{model_name} inference too slow"

@pytest.mark.integration
def test_model_preprocessing(ml_model_fixtures, sample_market_data):
    """Test preprocessing capabilities of models."""
    for model_name, model in ml_model_fixtures.items():
        stock_prices = sample_market_data['stock_prices']

        # Test preprocessing method
        preprocessed_data = model.preprocess(stock_prices)

        assert isinstance(preprocessed_data, torch.Tensor)
        assert preprocessed_data.min() >= 0
        assert preprocessed_data.max() <= 1
        assert preprocessed_data.device == model.device
