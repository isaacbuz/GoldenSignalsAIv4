"""
Metadata for pretrained models in GoldenSignalsAI.
"""
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PretrainedModelInfo:
    """Information about a pretrained model."""
    name: str
    type: str
    source: str
    trained_on: str
    last_updated: datetime
    input_shape: tuple
    output_shape: tuple
    description: str
    performance_metrics: Dict[str, float]

# Registry of pretrained models
PRETRAINED_MODELS = {
    "lstm_stock": PretrainedModelInfo(
        name="LSTM Stock Predictor",
        type="lstm",
        source="Stock-Prediction-Models",
        trained_on="S&P 500 stocks",
        last_updated=datetime(2024, 1, 1),
        input_shape=(60, 1),
        output_shape=(1,),
        description="LSTM model for stock price prediction",
        performance_metrics={
            "mse": 0.0023,
            "mae": 0.0456
        }
    ),
    "gru_stock": PretrainedModelInfo(
        name="GRU Stock Predictor",
        type="gru",
        source="Stock-Prediction-Models",
        trained_on="S&P 500 stocks",
        last_updated=datetime(2024, 1, 1),
        input_shape=(60, 1),
        output_shape=(1,),
        description="GRU model for stock price prediction",
        performance_metrics={
            "mse": 0.0021,
            "mae": 0.0445
        }
    ),
    "cnn_stock": PretrainedModelInfo(
        name="CNN Stock Predictor",
        type="cnn",
        source="Stock-Prediction-Models",
        trained_on="S&P 500 stocks",
        last_updated=datetime(2024, 1, 1),
        input_shape=(60, 1),
        output_shape=(1,),
        description="CNN model for stock price prediction",
        performance_metrics={
            "mse": 0.0025,
            "mae": 0.0478
        }
    ),
    "attention_stock": PretrainedModelInfo(
        name="Attention Stock Predictor",
        type="attention",
        source="Stock-Prediction-Models",
        trained_on="S&P 500 stocks",
        last_updated=datetime(2024, 1, 1),
        input_shape=(60, 1),
        output_shape=(1,),
        description="Attention-based model for stock price prediction",
        performance_metrics={
            "mse": 0.0020,
            "mae": 0.0432
        }
    )
}

def get_model_info(model_name: str) -> PretrainedModelInfo:
    """Get metadata for a pretrained model."""
    return PRETRAINED_MODELS.get(model_name)

def list_pretrained_models() -> Dict[str, PretrainedModelInfo]:
    """Get all available pretrained models."""
    return PRETRAINED_MODELS.copy()

def is_pretrained(model_name: str) -> bool:
    """Check if a model is pretrained."""
    return model_name in PRETRAINED_MODELS 