# Agent Communication Architecture - GoldenSignalsAI V2

## Overview

GoldenSignalsAI has evolved to support sophisticated inter-agent communication, allowing agents to share insights and build upon each other's analysis for more accurate trading signals.

## Current Agent Inventory

### Price Action & Pattern Recognition Agents
1. **Enhanced Pattern Agent** (`enhanced_pattern_agent.py`)
   - 15+ chart patterns (Head & Shoulders, Cup & Handle, Flags, etc.)
   - Multi-timeframe analysis
   - Smart Money divergence detection
   - Candlestick pattern recognition

2. **Legacy Price Action Agents** (archived, can be reactivated)
   - Candle Pattern Agent
   - Structure Break Agent

### Volume Analysis Agents
1. **Volume Spike Agent** - Detects unusual volume patterns
2. **Enhanced Volume Spike Agent** - With data sharing capabilities
3. **VWAP Agent** - Volume-weighted average price
4. **Volume Profile Agent** - Analyzes volume distribution at price levels
5. **Order Flow Agent** - Estimates buy/sell pressure

### Technical Indicators
- RSI, MACD, Moving Average Crossover
- Bollinger Bands, Stochastic, EMA
- Ichimoku, Fibonacci, ADX
- Parabolic SAR, Standard Deviation

### Market Analysis
- Market Profile Agent
- Sentiment Analysis Agent
- Options Flow Agent

### Meta Agents
- Simple Consensus Agent - Weighted voting
- ML Meta Agent - Dynamic weight optimization

## Communication Architecture

### 1. Current System (Indirect Communication)
```
Agent 1 → Signal → Orchestrator → Consensus Agent → Final Signal
Agent 2 → Signal ↗            ↘ ML Meta Agent
Agent 3 → Signal →              (learns from all)
```

### 2. Enhanced System (Direct Communication via Data Bus)

```
┌─────────────────────────────────────────────────┐
│              Agent Data Bus                      │
│  ┌─────────────────────────────────────────┐   │
│  │  Shared Context Store                    │   │
│  │  • Price Patterns                        │   │
│  │  • Volume Profiles                       │   │
│  │  • Support/Resistance                    │   │
│  │  • Market Regime                         │   │
│  │  • Order Flow                            │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
     ↑ Publish               ↓ Subscribe/Get
```

## Data Sharing Examples

### Volume Agent Sharing Insights
```python
# Volume agent detects spike and shares
volume_agent.publish_insight(
    "AAPL",
    SharedDataTypes.VOLUME_SPIKES,
    {
        'spike_type': 'bullish_spike',
        'volume_ratio': 3.5,
        'at_support': True
    }
)
```

### Pattern Agent Using Volume Context
```python
# Pattern agent gets volume context
context = pattern_agent.get_shared_context("AAPL", [
    SharedDataTypes.VOLUME_PROFILE,
    SharedDataTypes.ORDER_FLOW
])

# Makes decision based on pattern + volume
if pattern == "Bull Flag" and context['volume_profile']['poc'] < current_price:
    confidence += 0.2  # Pattern confirmed by volume
```

## Shared Data Types

### Price Action Data
- `SUPPORT_RESISTANCE`: Key price levels
- `TREND_DIRECTION`: Current trend state
- `PRICE_PATTERNS`: Detected chart patterns
- `KEY_LEVELS`: Important price points

### Volume Data
- `VOLUME_PROFILE`: Distribution of volume at prices
- `VOLUME_SPIKES`: Unusual volume activity
- `ORDER_FLOW`: Buy/sell pressure imbalance
- `ACCUMULATION_DISTRIBUTION`: Smart money activity

### Market Structure
- `MARKET_REGIME`: Trending/ranging, volatile/calm
- `VOLATILITY_STATE`: Current volatility conditions
- `LIQUIDITY_LEVELS`: Market depth information

### Sentiment & Flow
- `SENTIMENT_SCORE`: Market sentiment
- `OPTIONS_FLOW_BIAS`: Options activity bias
- `PUT_CALL_RATIO`: Options sentiment

## Implementation Benefits

### 1. **Confluence Detection**
Agents can confirm signals using multiple data sources:
- Volume spike + Support level + Bullish pattern = Higher confidence BUY

### 2. **Context-Aware Decisions**
- Pattern agent knows if volume confirms the pattern
- RSI agent knows if price is at key levels
- Volume agent knows the current trend direction

### 3. **Reduced False Signals**
- Agents filter signals based on peer insights
- ML Meta Agent learns which combinations work best

### 4. **Real-time Adaptation**
- Agents subscribe to updates and adjust in real-time
- Market regime changes propagate to all agents

## Usage Example

```python
from agents.common.data_bus import AgentDataBus
from agents.core.technical.enhanced_volume_spike_agent import EnhancedVolumeSpikeAgent

# Create shared infrastructure
data_bus = AgentDataBus()

# Initialize agents with data bus
volume_agent = EnhancedVolumeSpikeAgent(data_bus)
pattern_agent = EnhancedPatternAgent(data_bus)

# Agents automatically share insights
signal = volume_agent.generate_signal("AAPL")  # Publishes volume insights
signal = pattern_agent.generate_signal("AAPL")  # Uses volume context
```

## Future Enhancements

### 1. **Event-Driven Architecture**
- Market events trigger cascading analysis
- Agents react to specific conditions

### 2. **Agent Collaboration Patterns**
- Leader-follower patterns for confirmation
- Voting mechanisms for critical decisions

### 3. **Advanced ML Integration**
- Learn optimal agent communication patterns
- Discover new data relationships

### 4. **Performance Optimization**
- Selective data sharing based on relevance
- Caching frequently accessed insights

## Getting Started

1. **Enable Data Bus in Orchestrator**
```python
# In simple_orchestrator.py
self.data_bus = AgentDataBus()
```

2. **Upgrade Agents to Use Data Bus**
```python
# Convert existing agents to EnrichedAgent base class
class MyAgent(EnrichedAgent):
    def __init__(self, data_bus):
        super().__init__("MyAgent", data_bus)
```

3. **Define Data Sharing Strategy**
- What insights does each agent produce?
- What context does each agent need?
- How often should data be shared?

## Conclusion

The enhanced communication architecture transforms GoldenSignalsAI from a collection of independent agents into a collaborative AI trading system where agents build upon each other's insights for superior signal generation. 