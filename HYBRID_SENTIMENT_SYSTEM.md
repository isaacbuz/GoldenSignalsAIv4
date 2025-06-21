# GoldenSignalsAI V2 - Hybrid Sentiment System Documentation

## 🚀 Overview

The Hybrid Sentiment System represents a revolutionary approach to trading signal generation, combining independent agent analysis with collaborative intelligence. This system enhances the existing GoldenSignalsAI infrastructure with sophisticated sentiment analysis, divergence detection, and dynamic performance tracking.

## 🏗️ Architecture

### Core Components

1. **Agent Data Bus** (`agents/common/data_bus.py`)
   - Real-time publish/subscribe system
   - Time-based data expiration (TTL)
   - Thread-safe operations
   - Shared data types for price action, volume, market structure, and sentiment

2. **Hybrid Agent Base** (`agents/common/hybrid_agent_base.py`)
   - Dual signal generation (independent + collaborative)
   - Performance tracking for each approach
   - Divergence detection and scoring
   - Dynamic weight adjustment
   - Sentiment aggregation

3. **Hybrid Orchestrator** (`agents/orchestration/hybrid_orchestrator.py`)
   - Manages hybrid agents
   - Parallel execution
   - Divergence analysis
   - Performance dashboard
   - ML Meta Agent integration

4. **Enhanced ML Meta Agent** (`agents/meta/enhanced_ml_meta_agent.py`)
   - Ensemble optimization
   - Performance-based weight adjustment
   - Agent synergy detection

## 🤖 Hybrid Agents

### Implemented Agents

1. **HybridRSIAgent**
   - Independent: Pure RSI analysis
   - Collaborative: RSI with volume, pattern, and support/resistance context

2. **HybridVolumeAgent**
   - Independent: Volume spike detection
   - Collaborative: Volume with pattern and momentum confirmation

3. **HybridMACDAgent**
   - Independent: MACD crossovers
   - Collaborative: MACD with volume and trend alignment

4. **HybridBollingerAgent**
   - Independent: Band touches
   - Collaborative: Bands with volume and regime context

5. **HybridPatternAgent**
   - Independent: Chart pattern recognition
   - Collaborative: Patterns with volume and trend confirmation

### Signal Components

Each hybrid agent generates signals with three components:
```python
{
    "independent": {
        "action": "BUY/SELL/HOLD",
        "confidence": 0.0-1.0,
        "sentiment": "bullish/bearish/neutral"
    },
    "collaborative": {
        "action": "BUY/SELL/HOLD",
        "confidence": 0.0-1.0,
        "sentiment": "bullish/bearish/neutral"
    },
    "final": {
        "action": "BUY/SELL/HOLD",
        "confidence": 0.0-1.0,
        "sentiment": "strong_bullish/bullish/neutral/bearish/strong_bearish"
    }
}
```

## 📊 Data Sharing System

### SharedDataTypes

```python
{
    # Price Action Data
    "price_support_resistance": {"symbol": str, "levels": list, "timestamp": datetime},
    "price_trend_direction": {"symbol": str, "direction": str, "strength": float},
    "price_patterns": {"symbol": str, "patterns": list, "confidence": float},
    "price_key_levels": {"symbol": str, "levels": dict},
    
    # Volume Data
    "volume_profile": {"symbol": str, "profile": dict, "poc": float},
    "volume_spikes": {"symbol": str, "spikes": list, "significance": float},
    "volume_order_flow": {"symbol": str, "flow": dict, "imbalance": float},
    "volume_accumulation_distribution": {"symbol": str, "value": float, "trend": str},
    
    # Market Structure
    "market_regime": {"symbol": str, "regime": str, "confidence": float},
    "market_volatility_state": {"symbol": str, "state": str, "value": float},
    "market_liquidity_levels": {"symbol": str, "levels": dict},
    
    # Sentiment Data
    "sentiment_score": {"symbol": str, "score": float, "components": dict},
    "sentiment_options_flow": {"symbol": str, "bias": str, "magnitude": float},
    "sentiment_put_call_ratio": {"symbol": str, "ratio": float, "trend": str}
}
```

### Usage Example

```python
# Publishing data
agent.data_bus.publish('volume_spikes', {
    'symbol': 'AAPL',
    'spikes': [{'time': '10:30', 'volume': 1500000, 'price_change': 0.5}],
    'significance': 0.8,
    'timestamp': datetime.now()
})

# Subscribing to data
volume_data = agent.data_bus.get_latest('volume_spikes', 'AAPL')
```

## 🔄 Divergence Detection

### Types of Divergences

1. **Strong Divergence**: Independent and collaborative signals are opposite
   - Example: Independent=BUY, Collaborative=SELL
   - High contrarian opportunity potential

2. **Moderate Divergence**: Different signals but not opposite
   - Example: Independent=BUY, Collaborative=HOLD
   - Indicates uncertainty or transition

3. **Sentiment Divergence**: Same action but different sentiment
   - Example: Both BUY but Independent=bullish, Collaborative=strong_bullish
   - Shows conviction differences

### Divergence Scoring

```python
divergence_value = base_score * performance_multiplier * regime_factor

# Where:
# - base_score: 1.5 for strong, 1.2 for moderate, 1.0 for none
# - performance_multiplier: Historical success rate of divergences
# - regime_factor: Market regime suitability (1.3 in ranging, 0.8 in trending)
```

## 📈 Performance Tracking

### Metrics Tracked

1. **Independent Performance**
   - Accuracy rate
   - Average return
   - Win/loss ratio

2. **Collaborative Performance**
   - Accuracy rate
   - Average return
   - Improvement over independent

3. **Divergence Performance**
   - Success rate when divergence detected
   - Optimal divergence types by market regime

### Weight Adjustment

Dynamic weights adjust between 0.3 and 0.7 based on:
```python
new_weight = base_weight + learning_rate * (performance - baseline)

# Constraints:
independent_weight + collaborative_weight = 1.0
0.3 <= weight <= 0.7
```

## 🎭 Sentiment Analysis

### Sentiment Levels

1. **strong_bullish**: High conviction bullish (> 80% confidence)
2. **bullish**: Moderate bullish (60-80% confidence)
3. **neutral**: No clear direction (40-60% confidence)
4. **bearish**: Moderate bearish (20-40% confidence)
5. **strong_bearish**: High conviction bearish (< 20% confidence)

### Market Sentiment Aggregation

```python
market_sentiment = {
    "overall": weighted_average_sentiment,
    "confidence": average_confidence,
    "agreement": percentage_of_agents_in_agreement,
    "breakdown": {
        "bullish": count_of_bullish_agents,
        "bearish": count_of_bearish_agents,
        "neutral": count_of_neutral_agents
    }
}
```

## 🌐 API Endpoints

### Core Endpoints

1. **GET /api/v1/hybrid/signals/{symbol}**
   - Get hybrid signals for a specific symbol
   - Returns complete signal with sentiment and divergence analysis

2. **GET /api/v1/hybrid/sentiment/{symbol}**
   - Get sentiment analysis for symbol or overall market
   - Includes sentiment distribution and trends

3. **GET /api/v1/hybrid/performance**
   - Comprehensive performance dashboard
   - System health metrics and suggestions

4. **GET /api/v1/hybrid/divergences**
   - Analysis of current divergences across all symbols
   - Identifies high-opportunity symbols

5. **POST /api/v1/hybrid/performance/update**
   - Update agent performance with signal outcomes
   - Enables continuous learning

### WebSocket

**WS /api/v1/hybrid/ws**
- Real-time signal updates
- Sentiment changes
- Divergence alerts

## 🧪 Testing

### Test Suite

Run the comprehensive test suite:
```bash
python test_hybrid_system.py
```

Tests include:
1. Basic functionality
2. Divergence scenarios
3. Market sentiment evolution
4. Performance simulation
5. Comprehensive reporting

### Example Output

```
🎯 Final Signal: BUY (Confidence: 72.5%)
📝 Reasoning: Hybrid ensemble optimization

🎭 Market Sentiment: BULLISH
   Confidence: 68.3%
   Agreement: 75.0%

🔄 Divergence Analysis:
   Total Divergences: 2
   Strong Divergences: 1
   Opportunities: contrarian_momentum

🤖 Agent Breakdown:
   HybridRSIAgent:
   ├─ Final: BUY (65.0%)
   ├─ Independent: HOLD (45.0%) - neutral
   ├─ Collaborative: BUY (75.0%) - bullish
   ├─ ⚠️  Divergence: moderate
   └─ Reasoning: RSI approaching oversold with volume confirmation...
```

## 🔧 Configuration

### System Parameters

```python
# Hybrid Agent Configuration
INDEPENDENT_WEIGHT_RANGE = (0.3, 0.7)
LEARNING_RATE = 0.05
PERFORMANCE_WINDOW = 50  # signals
DIVERGENCE_BONUS_MAX = 0.2

# Data Bus Configuration
DEFAULT_TTL = 300  # seconds
MAX_HISTORY = 100  # items per data type

# Orchestrator Configuration
PARALLEL_EXECUTION = True
MAX_WORKERS = 10
SIGNAL_TIMEOUT = 5.0  # seconds
```

## 📊 Performance Optimization

### Best Practices

1. **Monitor Divergence Rates**
   - Healthy range: 10-30%
   - Too high: Agents may be misconfigured
   - Too low: Missing contrarian opportunities

2. **Balance Weight Updates**
   - Avoid rapid weight changes
   - Use smoothing factors
   - Consider market regime

3. **Optimize Data Sharing**
   - Only share relevant data
   - Use appropriate TTLs
   - Monitor bus performance

4. **Regular Performance Reviews**
   - Weekly weight analysis
   - Monthly divergence pattern review
   - Quarterly system optimization

## 🚀 Future Enhancements

1. **Advanced ML Integration**
   - Deep learning for pattern recognition
   - Reinforcement learning for weight optimization
   - Ensemble methods for signal combination

2. **Extended Data Sources**
   - Options flow integration
   - Social sentiment analysis
   - News impact scoring

3. **Risk Management**
   - Position sizing based on divergence
   - Dynamic stop-loss adjustment
   - Portfolio-level optimization

4. **Real-time Adaptation**
   - Intraday weight adjustments
   - Regime change detection
   - Adaptive learning rates

## 📝 Conclusion

The Hybrid Sentiment System represents a significant advancement in trading signal generation, combining the best of independent analysis with collaborative intelligence. By tracking performance, detecting divergences, and continuously adapting, the system provides robust and reliable trading signals while identifying unique market opportunities.

For questions or contributions, please refer to the main GoldenSignalsAI documentation. 