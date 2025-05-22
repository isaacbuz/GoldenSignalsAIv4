"""
Example: Using the AdvancedTradingStrategies registry
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategies.advanced_strategies import AdvancedTradingStrategies
import numpy as np

# Simulate some price data
prices = np.cumsum(np.random.normal(0, 1, 100))

print("Available strategies:", AdvancedTradingStrategies.available_strategies())

# Run a strategy by name
result = AdvancedTradingStrategies.run_strategy('momentum', prices)
print("Momentum signals:", result['signals'][:10])

# Try another
result = AdvancedTradingStrategies.run_strategy('volatility_breakout', prices)
print("Volatility breakout signals:", result['signals'][:10])
