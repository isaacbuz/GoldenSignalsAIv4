#!/usr/bin/env python3
"""
Demo of Chart Vision Analysis - AI analyzes trading chart screenshots
"""

import asyncio
import base64
from typing import Dict, Any


class ChartVisionDemo:
    """Demo version of chart vision analyzer"""
    
    async def analyze_chart(self, description: str, context: str = None) -> Dict[str, Any]:
        """Simulate analyzing a chart based on description"""
        
        # Simulate different chart scenarios
        if "bullish" in description.lower() or "uptrend" in description.lower():
            return self._analyze_bullish_chart(context)
        elif "bearish" in description.lower() or "downtrend" in description.lower():
            return self._analyze_bearish_chart(context)
        elif "triangle" in description.lower():
            return self._analyze_triangle_pattern(context)
        elif "support" in description.lower() or "resistance" in description.lower():
            return self._analyze_support_resistance(context)
        else:
            return self._analyze_general_chart(context)
    
    def _analyze_bullish_chart(self, context: str = None) -> Dict[str, Any]:
        """Analyze a bullish chart"""
        return {
            "analysis": """
## Chart Analysis Report

### Overview
I've analyzed your candlestick chart and identified several bullish characteristics.

### Trend Analysis
The chart shows a **bullish** trend. The upward momentum is strong with consistent higher highs and higher lows. This suggests continued buying pressure.

### Pattern Recognition

**Ascending Triangle**
- Confidence: 87%
- Target: $465.00
- Significance: Bullish continuation pattern with strong breakout potential

**Bull Flag**
- Confidence: 75%
- Target: $458.00
- Significance: Classic continuation pattern after strong upward move

### Key Levels
- **Resistance**: 450.00 (Strength: 85%, Touches: 4)
- **Support**: 442.50 (Strength: 90%, Touches: 5)
- **Support**: 438.00 (Strength: 75%, Touches: 3)

### Technical Indicators
- **Moving Average**: Price trading above 50 MA - bullish
- **RSI**: RSI at 65 - moderately bullish

### Key Insights
- 📈 Strong bullish momentum with RSI at 65 - still room to run
- 📐 Ascending triangle forming - breakout expected
- 🎯 Key resistance at 450.00 with 4 touches
- 📊 Price above 50 MA

### Trading Signals

**BUY Signal**
- Entry: 445.50
- Stop Loss: 442.00
- Take Profit: 458.00
- Confidence: 80%
- Reason: Ascending triangle pattern

### Context-Specific Analysis
""" + (self._get_bullish_context_analysis(context) if context else "Monitor the 450 resistance level for potential breakout opportunities."),
            
            "visual_data": {
                "chart_type": "candlestick",
                "detected_patterns": [
                    {
                        "type": "ascending_triangle",
                        "confidence": 0.87,
                        "target": 465.0
                    },
                    {
                        "type": "bull_flag",
                        "confidence": 0.75,
                        "target": 458.0
                    }
                ],
                "support_resistance": [
                    {"type": "resistance", "price": 450.0, "strength": 0.85},
                    {"type": "support", "price": 442.5, "strength": 0.90},
                    {"type": "support", "price": 438.0, "strength": 0.75}
                ],
                "trend": "bullish"
            },
            
            "trading_signals": [
                {
                    "type": "BUY",
                    "entry": 445.5,
                    "stop_loss": 442.0,
                    "take_profit": 458.0,
                    "confidence": 0.8,
                    "reason": "Ascending triangle pattern"
                }
            ],
            
            "confidence": 0.82
        }
    
    def _analyze_bearish_chart(self, context: str = None) -> Dict[str, Any]:
        """Analyze a bearish chart"""
        return {
            "analysis": """
## Chart Analysis Report

### Overview
I've analyzed your candlestick chart and identified bearish characteristics that warrant caution.

### Trend Analysis
The chart shows a **bearish** trend. The downward pressure is evident with lower highs and lower lows forming. Caution is advised for long positions.

### Pattern Recognition

**Head and Shoulders**
- Confidence: 85%
- Neckline: $445.50
- Target: $435.00
- Significance: Classic reversal pattern indicating potential further downside

**Descending Triangle**
- Confidence: 78%
- Support: $442.00
- Target: $438.00
- Significance: Bearish continuation pattern

### Key Levels
- **Resistance**: 448.00 (Strength: 90%, Touches: 5)
- **Resistance**: 446.00 (Strength: 85%, Touches: 3)
- **Support**: 442.00 (Strength: 75%, Touches: 4)

### Technical Indicators
- **Moving Average**: Price below 50 MA - bearish signal
- **RSI**: RSI at 35 - approaching oversold

### Key Insights
- 📉 Bearish trend with lower highs and lower lows
- 🎯 Head and Shoulders pattern detected - potential reversal at $445.50
- ⚠️ Price below 50 MA confirming bearish momentum
- 🔴 Support at 442.00 under pressure

### Trading Signals

**SELL Signal**
- Entry: 444.50
- Stop Loss: 446.50
- Take Profit: 438.00
- Confidence: 75%
- Reason: Head and shoulders pattern

### Risk Assessment
Elevated risk for long positions. Consider defensive strategies or wait for trend reversal confirmation.
""",
            
            "visual_data": {
                "chart_type": "candlestick",
                "detected_patterns": [
                    {
                        "type": "head_shoulders",
                        "confidence": 0.85,
                        "neckline": 445.5,
                        "target": 435.0
                    }
                ],
                "support_resistance": [
                    {"type": "resistance", "price": 448.0, "strength": 0.90},
                    {"type": "resistance", "price": 446.0, "strength": 0.85},
                    {"type": "support", "price": 442.0, "strength": 0.75}
                ],
                "trend": "bearish"
            },
            
            "trading_signals": [
                {
                    "type": "SELL",
                    "entry": 444.5,
                    "stop_loss": 446.5,
                    "take_profit": 438.0,
                    "confidence": 0.75,
                    "reason": "Head and shoulders pattern"
                }
            ],
            
            "confidence": 0.78
        }
    
    def _analyze_triangle_pattern(self, context: str = None) -> Dict[str, Any]:
        """Analyze triangle patterns"""
        return {
            "analysis": """
## Chart Analysis Report

### Overview
I've detected a triangle pattern formation in your chart, indicating potential breakout ahead.

### Pattern Recognition

**Symmetrical Triangle**
- Confidence: 82%
- Apex: Near $445.00
- Breakout Expected: Within 2-3 candles
- Target: $452.00 (upside) or $438.00 (downside)

### Key Observations
The converging trendlines suggest decreasing volatility and an imminent directional move. Volume has been declining within the pattern, which is typical. Watch for volume expansion on breakout.

### Trading Strategy
- Wait for confirmed breakout with volume
- Place stops just inside the triangle
- Target measured move equal to triangle height

### Key Levels
- Upper trendline resistance: $446.50
- Lower trendline support: $443.50
- Breakout confirmation: Close above $447.00 or below $443.00
""",
            
            "visual_data": {
                "chart_type": "candlestick",
                "detected_patterns": [
                    {
                        "type": "symmetrical_triangle",
                        "confidence": 0.82,
                        "apex": 445.0
                    }
                ],
                "trend": "neutral"
            },
            
            "confidence": 0.82
        }
    
    def _analyze_support_resistance(self, context: str = None) -> Dict[str, Any]:
        """Analyze support and resistance levels"""
        return {
            "analysis": """
## Chart Analysis Report

### Overview
I've identified key support and resistance levels that are crucial for trading decisions.

### Key Support & Resistance Levels

**Major Resistance Levels:**
- $450.00 - Strong resistance (tested 5 times)
- $447.50 - Intermediate resistance
- $445.00 - Minor resistance / previous support

**Major Support Levels:**
- $442.00 - Strong support (tested 6 times)
- $439.50 - Intermediate support
- $437.00 - Major support / potential bounce zone

### Trading Implications
- Current price near major support at $442.00
- Risk/Reward favors longs near support with stops below $441.50
- Resistance at $450.00 offers 8-point profit potential

### Volume Analysis
Higher volume on bounces from $442.00 support confirms buyer interest at this level.
""",
            
            "visual_data": {
                "support_resistance": [
                    {"type": "resistance", "price": 450.0, "strength": 0.95, "touches": 5},
                    {"type": "resistance", "price": 447.5, "strength": 0.70, "touches": 3},
                    {"type": "support", "price": 442.0, "strength": 0.90, "touches": 6},
                    {"type": "support", "price": 439.5, "strength": 0.75, "touches": 4}
                ]
            },
            
            "confidence": 0.85
        }
    
    def _analyze_general_chart(self, context: str = None) -> Dict[str, Any]:
        """General chart analysis"""
        return {
            "analysis": """
## Chart Analysis Report

### Overview
I've analyzed your chart and identified the following key characteristics:

### Market Structure
- Trend: Sideways consolidation
- Range: $442.00 - $448.00
- Current Position: Middle of range

### Technical Observations
- No clear directional bias
- Decreasing volume suggests consolidation
- Price respecting both support and resistance

### Trading Strategy
In ranging markets:
- Buy near support ($442.00)
- Sell near resistance ($448.00)
- Use tight stops due to range-bound action
- Wait for breakout for directional trades

### What to Watch
- Breakout above $448.00 targets $452.00
- Breakdown below $442.00 targets $438.00
- Volume expansion will confirm direction
""",
            
            "visual_data": {
                "chart_type": "candlestick",
                "trend": "sideways"
            },
            
            "confidence": 0.70
        }
    
    def _get_bullish_context_analysis(self, context: str) -> str:
        """Get context-specific analysis for bullish charts"""
        context_lower = context.lower()
        
        if "entry" in context_lower:
            return "Based on your interest in entry points, the best entry would be on a pullback to the $442.50 support level or on a confirmed breakout above $450.00 with volume. The ascending triangle pattern supports a bullish bias."
        elif "exit" in context_lower:
            return "For exit strategy, consider taking partial profits at $450.00 resistance and letting the rest run with a trailing stop. The pattern target of $465.00 offers significant upside."
        elif "risk" in context_lower:
            return "Risk is relatively low given the strong support at $442.50. A stop loss below $441.00 would limit downside to about 1%. The risk/reward ratio is favorable at approximately 1:3."
        else:
            return "The bullish setup is strong with multiple confirmations. Watch for volume on any breakout attempt above $450.00."


async def demo_chart_vision():
    """Run demo of chart vision analysis"""
    analyzer = ChartVisionDemo()
    
    print("🖼️  AI Chart Vision Analysis Demo")
    print("=" * 50)
    print("\nNote: In production, you would upload actual chart screenshots.")
    print("This demo simulates analysis based on chart descriptions.\n")
    
    # Demo scenarios
    scenarios = [
        {
            "description": "Bullish chart with ascending triangle pattern",
            "context": "Looking for entry points"
        },
        {
            "description": "Bearish chart with head and shoulders pattern",
            "context": "Should I exit my position?"
        },
        {
            "description": "Chart showing triangle pattern formation",
            "context": None
        },
        {
            "description": "Chart with clear support and resistance levels",
            "context": "What are the key levels to watch?"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['description']}")
        if scenario['context']:
            print(f"📝 Context: {scenario['context']}")
        print("-" * 50)
        
        # Analyze chart
        result = await analyzer.analyze_chart(
            scenario['description'],
            scenario['context']
        )
        
        # Print analysis
        print(result['analysis'])
        
        # Print detected patterns
        if 'visual_data' in result and 'detected_patterns' in result['visual_data']:
            print("\n🔍 Detected Patterns:")
            for pattern in result['visual_data']['detected_patterns']:
                print(f"  - {pattern['type'].replace('_', ' ').title()}: {pattern['confidence']:.0%} confidence")
        
        # Print trading signals
        if 'trading_signals' in result and result['trading_signals']:
            print("\n💡 Trading Signals:")
            for signal in result['trading_signals']:
                print(f"  - {signal['type']} @ {signal['entry']}")
                print(f"    Stop: {signal['stop_loss']}, Target: {signal['take_profit']}")
                print(f"    Confidence: {signal['confidence']:.0%}")
        
        print("\n" + "=" * 50)
        
        await asyncio.sleep(1)
    
    print("\n✅ Chart Vision Demo Complete!")
    print("\n🎯 In production, the AI can analyze:")
    print("  - Real chart screenshots (PNG, JPG, etc.)")
    print("  - Detect candlestick patterns")
    print("  - Identify support/resistance levels")
    print("  - Recognize chart patterns (triangles, H&S, flags, etc.)")
    print("  - Detect technical indicators if visible")
    print("  - Provide trading recommendations")
    print("  - Answer specific questions about the chart")
    print("\n📸 Simply attach a chart screenshot and ask your question!")


if __name__ == "__main__":
    asyncio.run(demo_chart_vision()) 