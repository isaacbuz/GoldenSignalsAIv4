# GoldenSignalsAI Agent Framework

## Directory Structure

The agent framework is organized into five main domains:

### 1. Core Trading Components (`core/`)
Contains the fundamental trading domains:
- `technical/`: Technical analysis agents (RSI, MACD, etc.)
- `fundamental/`: Fundamental analysis agents
- `sentiment/`: Market sentiment and psychology agents
- `risk/`: Risk management agents
- `timing/`: Market timing and execution agents
- `portfolio/`: Portfolio management agents

### 2. Infrastructure (`infrastructure/`)
System-level components:
- `data_sources/`: Data provider integrations and aggregation
- `integration/`: External system adapters and APIs
- `monitoring/`: System and performance monitoring

### 3. Research (`research/`)
Research and development components:
- `backtesting/`: Backtesting framework and utilities
- `ml/`: Machine learning models and predictive analysis
- `optimization/`: Strategy optimization and tuning

### 4. Common (`common/`)
Shared utilities and base components:
- `base/`: Base classes and interfaces
- `utils/`: Utility functions and helpers
- `registry/`: Agent registry and factory patterns

### 5. Experimental (`experimental/`)
Cutting-edge features and research:
- `grok/`: Grok-powered trading agents
- `vision/`: Computer vision-based analysis
- `adaptive/`: Adaptive learning systems

## Usage

Import components using their domain-specific paths:

```python
# Core components
from agents.core.technical import RSIAgent
from agents.core.fundamental import ValuationAgent

# Infrastructure
from agents.infrastructure.data_sources import DataSourceAgent

# Research
from agents.research.backtesting import BacktestResearchAgent

# Common utilities
from agents.common.base import BaseAgent
```

## Best Practices

1. Place new agents in their appropriate domain directory
2. Use the common utilities for shared functionality
3. Implement experimental features in the experimental directory
4. Follow the established patterns in each domain
5. Update the agent registry when adding new agents

## Contributing

When adding new components:
1. Choose the appropriate domain directory
2. Follow the existing naming conventions
3. Update the relevant __init__.py files
4. Add appropriate tests
5. Update documentation 