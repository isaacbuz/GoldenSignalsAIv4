# Technical Analysis Agents

## Overview
This directory contains agents specialized in technical analysis, focusing on price action, volume, and momentum indicators.

## Components

### Momentum Agents
- `RSIAgent`: Relative Strength Index analysis
- `MACDAgent`: Moving Average Convergence Divergence
- `RSIMACDAgent`: Combined RSI and MACD analysis
- `MomentumDivergenceAgent`: Price-momentum divergence detection

### Structure
```
technical/
├── momentum/        # Momentum-based indicators
├── volume/         # Volume analysis
├── price_action/   # Price pattern recognition
└── volatility/     # Volatility indicators
```

## Usage Examples

```python
from agents.core.technical import RSIAgent, MACDAgent

# Initialize RSI agent
rsi_agent = RSIAgent(
    period=14,
    overbought_threshold=70,
    oversold_threshold=30
)

# Initialize MACD agent
macd_agent = MACDAgent(
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```

## Best Practices
1. Use standardized input data formats (OHLCV)
2. Implement proper error handling for missing data
3. Include validation for indicator parameters
4. Document signal generation logic
5. Add unit tests for edge cases

## Signal Generation
Technical agents should implement:
- `generate_signal()`: Returns trading signals
- `calculate_indicator()`: Computes technical indicators
- `validate_data()`: Validates input data

## Performance Considerations
- Cache computed indicators when possible
- Use vectorized operations for calculations
- Consider using numba for intensive computations
- Implement proper memory management for large datasets 