# Adaptive Learning Architecture for GoldenSignalsAI

## Overview

The Adaptive Learning System enables agents to learn from backtest results and dynamically improve their signal generation accuracy. This creates a continuous feedback loop where agents evolve based on their performance.

## Core Components

### 1. Advanced Backtesting Engine (`src/domain/backtesting/advanced_backtest_engine.py`)
- **Purpose**: Comprehensive backtesting with agent attribution
- **Key Features**:
  - Agentic AI coordination
  - Multi-timeframe and multi-asset testing
  - Monte Carlo simulations (1000 runs)
  - Walk-forward analysis
  - Real-time and historical data support
  - Comprehensive metrics (Sharpe, Sortino, Calmar ratios)

### 2. Adaptive Learning System (`src/domain/backtesting/adaptive_learning_system.py`)
- **Purpose**: Analyze backtest results and generate learning recommendations
- **Key Features**:
  - Performance profiling by agent
  - Market regime analysis
  - Feature importance calculation
  - Confidence calibration
  - Meta-learning insights
  - Online model training

### 3. Adaptive Agent Interface (`src/agents/common/adaptive_agent_interface.py`)
- **Purpose**: Enable agents to receive and apply learning recommendations
- **Key Features**:
  - Dynamic configuration management
  - Recommendation processing
  - Performance tracking
  - State persistence
  - Exploration vs exploitation balance

## How It Works

### 1. Backtesting Phase
```python
# Run comprehensive backtest
engine = AdvancedBacktestEngine(config)
await engine.initialize()

results = await engine.run_backtest(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    initial_capital=100000
)
```

### 2. Learning Phase
```python
# Process results through learning system
learning_system = AdaptiveLearningSystem()
await learning_system.initialize()

learning_results = await learning_system.process_backtest_results(
    backtest_metrics=results['metrics'],
    trades=results['trades'],
    market_data=results['market_data']
)
```

### 3. Agent Adaptation Phase
```python
# Create adaptive agent
factory = AdaptiveAgentFactory(learning_system)
factory.register_agent(AdaptiveMomentumAgent, "momentum_agent_v2")

agent = await factory.create_agent("momentum_agent_v2")
# Agent automatically receives and applies recommendations
```

## Learning Feedback Loop

### 1. Trade Analysis
Each trade generates comprehensive feedback including:
- **Accuracy**: Was the trade profitable?
- **Reward**: Risk-adjusted return
- **Regret**: Opportunity cost
- **Surprise**: Prediction error

### 2. Performance Profiling
Agents are profiled across multiple dimensions:
- **By Market Regime**: Performance in different market conditions
- **By Volatility**: How volatility affects accuracy
- **By Time**: Intraday patterns
- **Feature Importance**: Which indicators drive success

### 3. Recommendation Generation
Based on analysis, specific recommendations are generated:

#### Parameter Adjustments
```json
{
  "type": "parameter_adjustment",
  "action": "increase_confidence_threshold",
  "reason": "Low accuracy: 42%",
  "suggested_value": 0.75
}
```

#### Model Recalibration
```json
{
  "type": "model_recalibration",
  "action": "recalibrate_confidence_scores",
  "calibration_data": {
    "0.6": {"expected": 0.6, "actual": 0.45},
    "0.7": {"expected": 0.7, "actual": 0.62}
  }
}
```

#### Regime-Specific Training
```json
{
  "type": "regime_specific_training",
  "action": "retrain_for_downtrend_high_vol",
  "reason": "Poor performance in downtrend_high_vol: 35%"
}
```

### 4. Agent Configuration Updates
Agents dynamically adjust their behavior:
- **Confidence thresholds** based on accuracy
- **Position sizing** based on drawdown
- **Feature selection** based on importance
- **Market regime filters** based on performance

## Database Schema

### Agent Profiles
```sql
CREATE TABLE agent_profiles (
    agent_id VARCHAR(100) PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    performance_data JSONB NOT NULL,
    model_parameters JSONB,
    learning_history JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Learning Feedback
```sql
CREATE TABLE learning_feedback (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(36) NOT NULL,
    agent_id VARCHAR(100) NOT NULL,
    signal_timestamp TIMESTAMPTZ NOT NULL,
    feedback_data JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE
);
```

### Model Versions
```sql
CREATE TABLE agent_model_versions (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    version INT NOT NULL,
    model_data BYTEA NOT NULL,
    performance_metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Meta-Learning Insights

The system identifies patterns across all agents:

### 1. Cross-Agent Patterns
- Successful trading patterns used by multiple agents
- Universal market conditions favorable for trading

### 2. Feature Universality
- Features that are important across multiple agents
- Consistency of feature importance

### 3. Ensemble Opportunities
- Complementary agent pairs
- Agents that perform well in different conditions

## Configuration Management

### Agent Configuration Structure
```python
@dataclass
class AgentConfiguration:
    # Core parameters
    confidence_threshold: float = 0.6
    position_size_multiplier: float = 1.0
    
    # Risk management
    stop_loss_percent: float = 0.02
    take_profit_percent: float = 0.05
    trailing_stop_enabled: bool = False
    
    # Feature selection
    enabled_features: List[str]
    feature_weights: Dict[str, float]
    
    # Market regime filters
    allowed_regimes: List[str]
    regime_adjustments: Dict[str, Dict[str, float]]
    
    # Learning parameters
    learning_enabled: bool = True
    exploration_rate: float = 0.1
```

## Implementation Example

### Creating an Adaptive Agent
```python
class AdaptiveMomentumAgent(AdaptiveAgentInterface):
    def __init__(self):
        super().__init__("momentum_agent_v2", "technical")
        
    async def generate_signals(self, market_data, positions):
        signals = []
        
        for symbol, df in market_data.items():
            # Calculate indicators
            momentum = self._calculate_momentum(df)
            
            # Apply learning adjustments
            raw_confidence = self._calculate_confidence(momentum)
            confidence = self.apply_confidence_calibration(raw_confidence)
            
            # Check regime filter
            regime = self._identify_regime(df)
            if not self.filter_by_regime(regime):
                continue
                
            # Generate signal with adaptive parameters
            if confidence >= self.config.confidence_threshold:
                signal = self._create_signal(symbol, df, confidence)
                signal = self.apply_regime_adjustment(signal, regime)
                signals.append(signal)
                
        return signals
```

## Performance Metrics

### Agent Performance Tracking
- **Accuracy**: Signal success rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Confidence Calibration**: How well confidence predicts success

### System-Wide Metrics
- **Overall Improvement**: Change in aggregate performance
- **Learning Efficiency**: How quickly agents adapt
- **Exploration Success**: Value of trying new strategies

## Best Practices

### 1. Gradual Adaptation
- Small incremental changes to avoid overfitting
- Maintain exploration to discover new patterns

### 2. Regime Awareness
- Different strategies for different market conditions
- Dynamic switching based on regime identification

### 3. Risk Management
- Tighter controls when performance degrades
- Position sizing based on confidence and volatility

### 4. Continuous Monitoring
- Track recent performance windows
- Adjust learning rates based on stability

## Future Enhancements

### 1. Deep Reinforcement Learning
- Neural network-based policy learning
- More complex state representations

### 2. Transfer Learning
- Share knowledge between similar agents
- Pre-trained models for new markets

### 3. Adversarial Training
- Robustness against market regime changes
- Stress testing with synthetic data

### 4. Ensemble Learning
- Dynamic agent weighting
- Automatic ensemble creation

## Conclusion

The Adaptive Learning System creates a powerful feedback loop that enables continuous improvement of trading agents. By analyzing backtest results, generating specific recommendations, and dynamically adjusting agent behavior, the system ensures that agents evolve to maintain optimal performance in changing market conditions. 