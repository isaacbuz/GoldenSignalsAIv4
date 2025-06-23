# Machine Learning Research Agents

## Overview
This directory contains machine learning models and predictive analysis agents for market forecasting and pattern recognition.

## Components

### Classifiers
- `EnsembleClassifierAgent`: Combines multiple classification models
- Base classifiers for market regime detection
- Specialized classifiers for pattern recognition

### Pretrained Models
- `BasePretrainedAgent`: Base class for pretrained model implementations
- `LSTMStockAgent`: LSTM-based stock price prediction
- Model metadata and versioning

### Forecasting
- `LSTMForecastAgent`: Time series forecasting using LSTM
- `ForecastEnsembleAgent`: Ensemble of multiple forecasting models

## Directory Structure
```
ml/
├── classifiers/           # Classification models
├── pretrained/           # Pre-trained model implementations
│   ├── models/          # Saved model files
│   └── metadata/        # Model metadata
├── forecasting/         # Time series forecasting
└── meta/               # Meta-learning and ensembles
```

## Model Management

### Model Registry
```python
from agents.research.ml import ModelRegistry

# Register a new model
registry = ModelRegistry()
registry.register_model(
    name="lstm_stock_predictor",
    version="1.0.0",
    metadata={
        "architecture": "LSTM",
        "input_features": ["price", "volume", "sentiment"],
        "output_features": ["price_prediction"]
    }
)
```

### Model Persistence
```python
from agents.common.utils.persistence import save_model, load_model

# Save trained model
save_model(model, "lstm_stock_predictor", version="1.0.0")

# Load model
model = load_model("lstm_stock_predictor", version="1.0.0")
```

## Best Practices
1. Version all models and maintain metadata
2. Implement proper train/validation/test splits
3. Use proper cross-validation for time series
4. Document model architectures and hyperparameters
5. Maintain reproducible training pipelines
6. Implement proper early stopping and model checkpointing
7. Use proper evaluation metrics for financial data

## Performance Monitoring
- Track prediction accuracy over time
- Monitor for concept drift
- Implement automatic retraining triggers
- Log prediction confidence scores

## Dependencies
Required packages:
- tensorflow>=2.6.0
- torch>=1.9.0
- scikit-learn>=0.24.2
- pandas>=1.3.0
- numpy>=1.19.5 