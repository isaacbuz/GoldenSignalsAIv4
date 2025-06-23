# GoldenSignalsAI Visualization Features Guide

## Overview

This guide covers the advanced visualization features implemented in GoldenSignalsAI, including candlestick pattern recognition and predictive trend visualization.

## Features Implemented

### 1. Candlestick Pattern Recognition

#### Supported Patterns

**Single Candle Patterns:**
- Doji (55% success rate)
- Hammer (62% success rate)
- Inverted Hammer (60% success rate)
- Shooting Star (59% success rate)
- Hanging Man (57% success rate)
- Spinning Top (54% success rate)
- Marubozu (71% success rate)
- Long-legged Doji (56% success rate)
- Dragonfly Doji (58% success rate)
- Gravestone Doji (57% success rate)

**Two Candle Patterns:**
- Bullish Engulfing (65% success rate)
- Bearish Engulfing (72% success rate)
- Tweezer Top (58% success rate)
- Tweezer Bottom (60% success rate)
- Piercing Line (64% success rate)
- Dark Cloud Cover (66% success rate)
- Bullish Harami (61% success rate)
- Bearish Harami (63% success rate)

**Three Candle Patterns:**
- Morning Star (65% success rate)
- Evening Star (69% success rate)
- Three White Soldiers (82% success rate)
- Three Black Crows (78% success rate)
- Three Inside Up (65% success rate)
- Three Inside Down (67% success rate)

**Complex Patterns:**
- Rising Three Methods (74% success rate)
- Falling Three Methods (72% success rate)
- Mat Hold (70% success rate)

### 2. Predictive Trend Visualization

#### Features:
- Multi-model ensemble predictions (Linear, Polynomial, Technical, ML, Monte Carlo)
- Confidence intervals with upper and lower bounds
- Support and resistance level detection
- Momentum and volatility scoring
- Trend direction and strength analysis
- Key price level identification

### 3. Integrated Analysis

Combines pattern recognition with predictive analytics to provide:
- Signal alignment verification
- Trading recommendations
- Risk/reward calculations
- Entry, target, and stop-loss levels

## API Endpoints

### 1. Get Candlestick Patterns

```bash
GET /api/v1/patterns/{symbol}?lookback=100
```

**Response:**
```json
{
  "symbol": "AAPL",
  "patterns": [
    {
      "type": "hammer",
      "timestamp": "2024-01-19T10:00:00Z",
      "price": 152.50,
      "direction": "bullish",
      "strength": 85.0,
      "confidence": 78.5,
      "successRate": 0.62,
      "description": "Bullish reversal pattern with small body at top and long lower shadow",
      "targets": {
        "priceTarget": 155.20,
        "stopLoss": 151.00
      }
    }
  ],
  "statistics": {
    "total_patterns": 25,
    "bullish_count": 10,
    "bearish_count": 8,
    "neutral_count": 7,
    "avg_confidence": 72.5,
    "avg_success_rate": 0.65
  }
}
```

### 2. Get Price Predictions

```bash
GET /api/v1/predictions/{symbol}?timeframe=1h&periods=20
```

**Parameters:**
- `timeframe`: 5m, 15m, 30m, 1h, 4h, 1d, 1w
- `periods`: Number of prediction periods (default: 20)

**Response:**
```json
{
  "symbol": "AAPL",
  "currentPrice": 155.50,
  "predictions": [
    {
      "timestamp": "2024-01-19T11:00:00Z",
      "price": 156.20,
      "confidence": 95.2,
      "upperBound": 157.50,
      "lowerBound": 154.90
    }
  ],
  "trend": {
    "direction": "bullish",
    "strength": 78.5,
    "confidence": 82.3
  },
  "levels": {
    "support": [152.30, 150.50, 148.00],
    "resistance": [158.50, 160.00, 162.50],
    "key": [
      {
        "price": 152.30,
        "type": "support",
        "strength": 85,
        "distance": 2.1
      }
    ]
  },
  "metrics": {
    "momentum": 35.2,
    "volatility": 18.5
  }
}
```

## Frontend Components

### PredictiveTradingChart

Enhanced trading chart component with integrated visualization features.

```tsx
import PredictiveTradingChart from './components/Chart/PredictiveTradingChart';

<PredictiveTradingChart
  symbol="AAPL"
  height={600}
  onPatternDetected={(pattern) => console.log('Pattern detected:', pattern)}
  onPredictionUpdate={(prediction) => console.log('Prediction updated:', prediction)}
/>
```

**Features:**
- Real-time candlestick pattern markers
- Prediction trend lines with confidence bands
- Support/resistance level visualization
- Interactive zoom and pan controls
- Pattern and prediction info panels

## Usage Examples

### 1. Basic Pattern Detection

```python
from src.services.candlestick_patterns import CandlestickPatternService

# Initialize service
pattern_service = CandlestickPatternService()

# Detect patterns
patterns = pattern_service.detect_all_patterns(df, lookback=100)

# Get statistics
stats = pattern_service.get_pattern_statistics(patterns)
```

### 2. Generate Predictions

```python
from src.services.prediction_visualization import (
    PredictionVisualizationService, 
    PredictionTimeframe
)

# Initialize service
prediction_service = PredictionVisualizationService()

# Generate predictions
prediction = prediction_service.generate_predictions(
    symbol="AAPL",
    historical_data=df,
    timeframe=PredictionTimeframe.HOUR_4,
    prediction_periods=20
)

print(f"Trend: {prediction.trend_direction}")
print(f"Target: ${prediction.prediction_points[-1].price:.2f}")
```

### 3. Integrated Analysis

```python
# Combine patterns and predictions
patterns = pattern_service.detect_all_patterns(df)
prediction = prediction_service.generate_predictions(symbol, df, timeframe)

# Check alignment
bullish_patterns = [p for p in patterns if p.direction == 'bullish']
bearish_patterns = [p for p in patterns if p.direction == 'bearish']

if len(bullish_patterns) > len(bearish_patterns) and prediction.trend_direction == 'bullish':
    print("Strong bullish signal - patterns and predictions aligned")
```

## Technical Implementation

### Pattern Recognition Algorithm

1. **TA-Lib Integration**: Uses industry-standard technical analysis library
2. **Custom Pattern Detection**: Implements additional patterns not in TA-Lib
3. **Confidence Scoring**: Based on volume confirmation, trend alignment, and location
4. **Success Rate Tracking**: Historical success rates for each pattern type

### Prediction Models

1. **Linear Regression**: Basic trend projection
2. **Polynomial Regression**: Captures non-linear trends
3. **Technical Analysis**: Uses indicators for prediction
4. **ML Ensemble**: Random Forest with feature engineering
5. **Monte Carlo**: Probabilistic simulation (1000 runs)

### Visualization Features

1. **Lightweight Charts**: High-performance charting library
2. **Real-time Updates**: WebSocket integration for live data
3. **Interactive Markers**: Pattern indicators on chart
4. **Confidence Bands**: Visual representation of prediction uncertainty
5. **Multi-layer Display**: Overlays for patterns, predictions, and levels

## Best Practices

### 1. Pattern Recognition

- Use longer lookback periods (50-100 candles) for better accuracy
- Combine multiple patterns for stronger signals
- Consider volume confirmation for pattern validation
- Check pattern location relative to support/resistance

### 2. Prediction Visualization

- Shorter timeframes (5m, 15m) have lower confidence
- Use ensemble predictions for better accuracy
- Monitor prediction confidence decay over time
- Combine with other indicators for confirmation

### 3. Trading Decisions

- Never rely on single indicator
- Wait for pattern and prediction alignment
- Use proper risk management (stop-loss)
- Consider market conditions and volatility

## Performance Considerations

### Optimization Tips

1. **Data Caching**: Results cached for 5 minutes
2. **Batch Processing**: Process multiple symbols together
3. **Selective Updates**: Only update changed data
4. **Efficient Algorithms**: O(n) pattern detection

### Resource Usage

- Pattern detection: ~50ms for 100 candles
- Prediction generation: ~200ms for 20 periods
- Memory usage: ~10MB per symbol
- API rate limits: 100 requests/minute

## Troubleshooting

### Common Issues

1. **No patterns detected**
   - Check data quality and completeness
   - Ensure sufficient historical data (min 20 candles)
   - Verify column names are lowercase

2. **Prediction errors**
   - Ensure data has no gaps
   - Check for sufficient data points
   - Verify timeframe is supported

3. **Visualization not updating**
   - Check WebSocket connection
   - Verify API endpoints are accessible
   - Clear browser cache

## Future Enhancements

1. **Additional Patterns**
   - Harmonic patterns (Gartley, Butterfly)
   - Chart patterns (Head & Shoulders, Triangles)
   - Custom pattern definition

2. **Advanced Predictions**
   - LSTM neural networks
   - Transformer models
   - Multi-timeframe analysis

3. **Enhanced Visualization**
   - 3D price projections
   - Heatmap overlays
   - Pattern success statistics

## Conclusion

The visualization features in GoldenSignalsAI provide powerful tools for technical analysis and predictive trading. By combining traditional candlestick pattern recognition with modern machine learning predictions, traders can make more informed decisions with visual confirmation of signals.

For questions or support, please refer to the main documentation or contact the development team. 