"""
Test script for config-driven, registry-based model and strategy selection in GoldenSignalsAI.
"""
import yaml
import pandas as pd
import numpy as np
from strategies.advanced_strategies import AdvancedStrategies
from domain.models.factory import ModelFactory

# Load config
with open('config/parameters.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Example: test strategy registry
print('Available strategies:', AdvancedStrategies.available_strategies())
strat_name = 'moving_average_crossover'
strat_params = config['strategies'][strat_name]
# Simulate price data
np.random.seed(42)
df = pd.DataFrame({'Close': np.cumsum(np.random.randn(100)) + 100})
signal = AdvancedStrategies.run_strategy(strat_name, df, **strat_params)
print(f"Signal from {strat_name}:\n", signal.tail())

# Example: test model registry
print('Available models:', ModelFactory.available_models())
model_name = 'lstm'
model_params = config['models'][model_name]
model = ModelFactory.get_model(model_name, config=model_params)
# Simulate data for fit/predict
X = np.random.randn(100, model_params.get('sequence_length', 50), model_params.get('features', 1))
y = np.random.randn(100)
try:
    model.fit(X, y)
    preds = model.predict(X)
    print(f"Predictions from {model_name}:\n", preds[:5])
except Exception as e:
    print(f"Model error: {e}")
