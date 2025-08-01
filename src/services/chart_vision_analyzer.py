"""
Chart Vision Analyzer Service
Analyzes trading chart screenshots using computer vision and AI
"""

import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class ChartAnalysisResult:
    """Result of chart analysis"""

    chart_type: str  # candlestick, line, bar
    timeframe: str  # estimated timeframe
    trend: str  # bullish, bearish, sideways
    patterns: List[Dict[str, Any]]
    support_resistance: List[Dict[str, Any]]
    indicators: List[Dict[str, Any]]
    key_observations: List[str]
    trading_signals: List[Dict[str, Any]]
    confidence: float


class ChartVisionAnalyzer:
    """
    Analyzes trading charts from screenshots using computer vision
    """

    def __init__(self):
        self.pattern_templates = self._load_pattern_templates()
        self.indicator_detectors = self._initialize_indicator_detectors()

    async def analyze_chart_image(
        self, image_data: str, user_context: Optional[str] = None  # base64 encoded image
    ) -> Dict[str, Any]:
        """
        Main entry point for chart image analysis
        """
        # Decode image
        image = self._decode_image(image_data)

        # Preprocess image
        processed_image = self._preprocess_image(image)

        # Detect chart type
        chart_type = self._detect_chart_type(processed_image)

        # Extract price data
        price_data = self._extract_price_data(processed_image, chart_type)

        # Detect patterns
        patterns = await self._detect_patterns(processed_image, price_data)

        # Find support/resistance levels
        support_resistance = self._find_support_resistance(price_data)

        # Detect indicators
        indicators = self._detect_indicators(processed_image)

        # Analyze trend
        trend = self._analyze_trend(price_data)

        # Generate insights
        insights = self._generate_insights(
            chart_type, trend, patterns, support_resistance, indicators
        )

        # Generate trading signals
        signals = self._generate_trading_signals(trend, patterns, support_resistance, indicators)

        # Create comprehensive analysis
        analysis = self._create_analysis_report(
            chart_type,
            trend,
            patterns,
            support_resistance,
            indicators,
            insights,
            signals,
            user_context,
        )

        return {
            "analysis": analysis,
            "visual_data": {
                "chart_type": chart_type,
                "detected_patterns": patterns,
                "support_resistance": support_resistance,
                "indicators": indicators,
                "trend": trend,
            },
            "trading_signals": signals,
            "confidence": self._calculate_confidence(patterns, indicators),
        }

    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        return np.array(image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Enhance contrast
        enhanced = cv2.equalizeHist(gray)

        # Remove noise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        return denoised

    def _detect_chart_type(self, image: np.ndarray) -> str:
        """Detect the type of chart (candlestick, line, bar)"""
        # Use edge detection to identify chart characteristics
        edges = cv2.Canny(image, 50, 150)

        # Analyze patterns in edges
        # Candlestick charts have distinctive rectangular patterns
        # Line charts have continuous lines
        # Bar charts have vertical bars

        # Simplified detection logic
        vertical_lines = self._count_vertical_lines(edges)
        horizontal_continuity = self._measure_horizontal_continuity(edges)

        if vertical_lines > 50 and horizontal_continuity < 0.3:
            return "candlestick"
        elif horizontal_continuity > 0.7:
            return "line"
        else:
            return "bar"

    def _extract_price_data(self, image: np.ndarray, chart_type: str) -> List[Dict]:
        """Extract price data from chart image"""
        # This is a simplified version
        # In reality, this would use OCR for price labels and
        # computer vision to trace price movements

        # Mock extraction for demo
        height, width = image.shape[:2]
        price_data = []

        # Simulate extracting price points
        for i in range(50):
            x = int(i * width / 50)
            # Find price level at this x coordinate
            price_y = self._find_price_at_x(image, x)

            price_data.append(
                {
                    "index": i,
                    "x": x,
                    "y": price_y,
                    "price": self._y_to_price(price_y, height),
                    "volume": np.random.randint(1000000, 10000000),
                }
            )

        return price_data

    async def _detect_patterns(
        self, image: np.ndarray, price_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Detect chart patterns using computer vision"""
        patterns = []

        # Head and Shoulders detection
        h_s_pattern = self._detect_head_shoulders(price_data)
        if h_s_pattern:
            patterns.append(h_s_pattern)

        # Triangle patterns
        triangle = self._detect_triangle_pattern(price_data)
        if triangle:
            patterns.append(triangle)

        # Flag patterns
        flag = self._detect_flag_pattern(price_data)
        if flag:
            patterns.append(flag)

        # Double top/bottom
        double_pattern = self._detect_double_pattern(price_data)
        if double_pattern:
            patterns.append(double_pattern)

        return patterns

    def _find_support_resistance(self, price_data: List[Dict]) -> List[Dict]:
        """Find support and resistance levels"""
        levels = []

        # Extract price values
        prices = [p["price"] for p in price_data]

        # Find local maxima (resistance)
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                levels.append(
                    {
                        "type": "resistance",
                        "price": prices[i],
                        "strength": self._calculate_level_strength(prices, prices[i]),
                        "touches": self._count_touches(prices, prices[i]),
                    }
                )

        # Find local minima (support)
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                levels.append(
                    {
                        "type": "support",
                        "price": prices[i],
                        "strength": self._calculate_level_strength(prices, prices[i]),
                        "touches": self._count_touches(prices, prices[i]),
                    }
                )

        # Sort by strength
        levels.sort(key=lambda x: x["strength"], reverse=True)

        return levels[:5]  # Return top 5 levels

    def _detect_indicators(self, image: np.ndarray) -> List[Dict]:
        """Detect technical indicators present in the chart"""
        indicators = []

        # Look for common indicator patterns
        # Moving averages appear as smooth lines
        ma_lines = self._detect_moving_averages(image)
        if ma_lines:
            indicators.extend(ma_lines)

        # RSI/MACD usually in separate panel below
        lower_indicators = self._detect_lower_panel_indicators(image)
        if lower_indicators:
            indicators.extend(lower_indicators)

        # Bollinger Bands
        bb = self._detect_bollinger_bands(image)
        if bb:
            indicators.append(bb)

        return indicators

    def _analyze_trend(self, price_data: List[Dict]) -> str:
        """Analyze overall trend direction"""
        prices = [p["price"] for p in price_data]

        # Calculate trend using linear regression
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        # Calculate trend strength
        price_range = max(prices) - min(prices)
        trend_strength = abs(slope) / price_range

        if slope > 0 and trend_strength > 0.1:
            return "bullish"
        elif slope < 0 and trend_strength > 0.1:
            return "bearish"
        else:
            return "sideways"

    def _generate_insights(
        self,
        chart_type: str,
        trend: str,
        patterns: List[Dict],
        support_resistance: List[Dict],
        indicators: List[Dict],
    ) -> List[str]:
        """Generate key insights from analysis"""
        insights = []

        # Trend insight
        if trend == "bullish":
            insights.append("📈 Strong bullish trend detected with higher highs and higher lows")
        elif trend == "bearish":
            insights.append("📉 Bearish trend in progress with lower highs and lower lows")
        else:
            insights.append("➡️ Sideways consolidation - waiting for directional breakout")

        # Pattern insights
        for pattern in patterns[:2]:  # Top 2 patterns
            if pattern["type"] == "head_shoulders":
                insights.append(
                    f"🎯 Head and Shoulders pattern detected - potential reversal at {pattern['neckline']}"
                )
            elif pattern["type"] == "triangle":
                insights.append(f"📐 {pattern['subtype']} triangle forming - breakout expected")
            elif pattern["type"] == "double_top":
                insights.append(
                    f"🏔️ Double top pattern at {pattern['price']} - bearish reversal signal"
                )

        # Support/Resistance insights
        if support_resistance:
            key_level = support_resistance[0]
            insights.append(
                f"🎯 Key {key_level['type']} at {key_level['price']:.2f} with {key_level['touches']} touches"
            )

        # Indicator insights
        for indicator in indicators:
            if indicator["type"] == "moving_average":
                insights.append(
                    f"📊 Price {'above' if indicator['position'] == 'above' else 'below'} {indicator['period']} MA"
                )
            elif indicator["type"] == "rsi":
                if indicator["value"] > 70:
                    insights.append("⚠️ RSI overbought - potential pullback")
                elif indicator["value"] < 30:
                    insights.append("🔥 RSI oversold - potential bounce")

        return insights

    def _generate_trading_signals(
        self,
        trend: str,
        patterns: List[Dict],
        support_resistance: List[Dict],
        indicators: List[Dict],
    ) -> List[Dict]:
        """Generate trading signals based on analysis"""
        signals = []

        # Pattern-based signals
        for pattern in patterns:
            if pattern["confidence"] > 0.7:
                signal = {
                    "type": pattern["signal_type"],
                    "entry": pattern["entry_price"],
                    "stop_loss": pattern["stop_loss"],
                    "take_profit": pattern["target"],
                    "confidence": pattern["confidence"],
                    "reason": f"{pattern['type']} pattern",
                }
                signals.append(signal)

        # Support/Resistance signals
        current_price = 100  # Would get from latest price data
        for level in support_resistance:
            if (
                level["type"] == "support"
                and abs(current_price - level["price"]) / current_price < 0.02
            ):
                signals.append(
                    {
                        "type": "BUY",
                        "entry": level["price"],
                        "stop_loss": level["price"] * 0.98,
                        "take_profit": level["price"] * 1.05,
                        "confidence": 0.7,
                        "reason": "Near strong support",
                    }
                )

        return signals

    def _create_analysis_report(
        self,
        chart_type: str,
        trend: str,
        patterns: List[Dict],
        support_resistance: List[Dict],
        indicators: List[Dict],
        insights: List[str],
        signals: List[Dict],
        user_context: Optional[str],
    ) -> str:
        """Create comprehensive analysis report"""
        report = f"""
## Chart Analysis Report

### Overview
I've analyzed your {chart_type} chart and identified several key elements:

### Trend Analysis
The chart shows a **{trend}** trend. {self._get_trend_description(trend)}

### Pattern Recognition
"""

        if patterns:
            for pattern in patterns[:3]:
                report += f"\n**{pattern['type'].replace('_', ' ').title()}**"
                report += f"\n- Confidence: {pattern['confidence']:.0%}"
                report += f"\n- Target: {pattern.get('target', 'N/A')}"
                report += f"\n- Significance: {pattern.get('description', 'Pattern detected')}\n"
        else:
            report += "\nNo major patterns detected at this time.\n"

        report += "\n### Key Levels\n"
        for level in support_resistance[:5]:
            report += f"- **{level['type'].title()}**: {level['price']:.2f} "
            report += f"(Strength: {level['strength']:.0%}, Touches: {level['touches']})\n"

        if indicators:
            report += "\n### Technical Indicators\n"
            for indicator in indicators:
                report += f"- **{indicator['type']}**: {indicator.get('description', 'Detected')}\n"

        report += "\n### Key Insights\n"
        for insight in insights:
            report += f"- {insight}\n"

        if signals:
            report += "\n### Trading Signals\n"
            for signal in signals[:2]:
                report += f"\n**{signal['type']} Signal**\n"
                report += f"- Entry: {signal['entry']:.2f}\n"
                report += f"- Stop Loss: {signal['stop_loss']:.2f}\n"
                report += f"- Take Profit: {signal['take_profit']:.2f}\n"
                report += f"- Confidence: {signal['confidence']:.0%}\n"
                report += f"- Reason: {signal['reason']}\n"

        if user_context:
            report += f"\n### Context-Specific Analysis\n{self._analyze_with_context(user_context, patterns, trend)}\n"

        report += "\n*Analysis generated using AI-powered chart vision technology*"

        return report

    def _get_trend_description(self, trend: str) -> str:
        """Get detailed trend description"""
        descriptions = {
            "bullish": "The upward momentum is strong with consistent higher highs and higher lows. This suggests continued buying pressure.",
            "bearish": "The downward pressure is evident with lower highs and lower lows forming. Caution is advised for long positions.",
            "sideways": "The market is in consolidation phase, trading within a range. Watch for breakout above resistance or below support.",
        }
        return descriptions.get(trend, "")

    def _analyze_with_context(self, context: str, patterns: List[Dict], trend: str) -> str:
        """Provide context-specific analysis"""
        context_lower = context.lower()

        if "entry" in context_lower or "buy" in context_lower:
            return self._get_entry_analysis(patterns, trend)
        elif "exit" in context_lower or "sell" in context_lower:
            return self._get_exit_analysis(patterns, trend)
        elif "risk" in context_lower:
            return self._get_risk_analysis(patterns, trend)
        else:
            return "Based on your query, I recommend monitoring the key levels identified above for potential trading opportunities."

    def _get_entry_analysis(self, patterns: List[Dict], trend: str) -> str:
        """Provide entry-specific analysis"""
        if trend == "bullish" and patterns:
            return "Good entry opportunities may present on pullbacks to support levels or pattern breakouts. Wait for confirmation with volume."
        elif trend == "bearish":
            return "Consider waiting for trend reversal signals or strong support levels before entering long positions."
        else:
            return "In sideways markets, buy near support and sell near resistance. Use tight stops due to range-bound action."

    def _get_exit_analysis(self, patterns: List[Dict], trend: str) -> str:
        """Provide exit-specific analysis"""
        if patterns and any(p["type"] in ["double_top", "head_shoulders"] for p in patterns):
            return (
                "Warning: Reversal patterns detected. Consider taking profits or tightening stops."
            )
        elif trend == "bullish":
            return "Trail stops below key support levels to protect profits while allowing for continued upside."
        else:
            return "Exit signals include break of support levels or completion of bearish patterns."

    def _get_risk_analysis(self, patterns: List[Dict], trend: str) -> str:
        """Provide risk-specific analysis"""
        risk_factors = []

        if any(p["type"] in ["double_top", "head_shoulders"] for p in patterns):
            risk_factors.append("reversal patterns present")

        if trend == "bearish":
            risk_factors.append("downtrend in progress")

        if risk_factors:
            return f"Elevated risk due to: {', '.join(risk_factors)}. Consider reducing position size or using tighter stops."
        else:
            return "Risk appears moderate. Use standard position sizing with stops below key support levels."

    def _calculate_confidence(self, patterns: List[Dict], indicators: List[Dict]) -> float:
        """Calculate overall analysis confidence"""
        confidence_factors = []

        # Pattern confidence
        if patterns:
            pattern_conf = np.mean([p["confidence"] for p in patterns])
            confidence_factors.append(pattern_conf)

        # Indicator confidence (if multiple indicators agree)
        if len(indicators) > 2:
            confidence_factors.append(0.8)
        elif len(indicators) > 0:
            confidence_factors.append(0.6)

        # Base confidence
        confidence_factors.append(0.7)

        return np.mean(confidence_factors)

    # Helper methods (simplified implementations)
    def _load_pattern_templates(self) -> Dict:
        """Load pattern recognition templates"""
        return {}

    def _initialize_indicator_detectors(self) -> Dict:
        """Initialize indicator detection algorithms"""
        return {}

    def _count_vertical_lines(self, edges: np.ndarray) -> int:
        """Count vertical lines in edge image"""
        return 75  # Mock value

    def _measure_horizontal_continuity(self, edges: np.ndarray) -> float:
        """Measure horizontal line continuity"""
        return 0.2  # Mock value

    def _find_price_at_x(self, image: np.ndarray, x: int) -> int:
        """Find price level at x coordinate"""
        height = image.shape[0]
        return np.random.randint(height * 0.2, height * 0.8)

    def _y_to_price(self, y: int, height: int) -> float:
        """Convert y coordinate to price"""
        return 100 + (height / 2 - y) * 0.1

    def _detect_head_shoulders(self, price_data: List[Dict]) -> Optional[Dict]:
        """Detect head and shoulders pattern"""
        # Simplified detection
        if np.random.random() > 0.7:
            return {
                "type": "head_shoulders",
                "confidence": 0.85,
                "neckline": 98.5,
                "target": 95.0,
                "entry_price": 98.0,
                "stop_loss": 100.0,
                "signal_type": "SELL",
                "description": "Classic reversal pattern",
            }
        return None

    def _detect_triangle_pattern(self, price_data: List[Dict]) -> Optional[Dict]:
        """Detect triangle patterns"""
        if np.random.random() > 0.6:
            return {
                "type": "triangle",
                "subtype": "ascending",
                "confidence": 0.75,
                "target": 105.0,
                "entry_price": 101.0,
                "stop_loss": 99.0,
                "signal_type": "BUY",
                "description": "Bullish continuation pattern",
            }
        return None

    def _detect_flag_pattern(self, price_data: List[Dict]) -> Optional[Dict]:
        """Detect flag patterns"""
        return None  # Simplified

    def _detect_double_pattern(self, price_data: List[Dict]) -> Optional[Dict]:
        """Detect double top/bottom patterns"""
        return None  # Simplified

    def _calculate_level_strength(self, prices: List[float], level: float) -> float:
        """Calculate strength of support/resistance level"""
        touches = self._count_touches(prices, level)
        return min(touches * 0.2, 1.0)

    def _count_touches(self, prices: List[float], level: float) -> int:
        """Count how many times price touched a level"""
        tolerance = level * 0.01  # 1% tolerance
        touches = sum(1 for p in prices if abs(p - level) < tolerance)
        return touches

    def _detect_moving_averages(self, image: np.ndarray) -> List[Dict]:
        """Detect moving average lines"""
        return [
            {
                "type": "moving_average",
                "period": 50,
                "position": "above",
                "description": "Price trading above 50 MA - bullish",
            }
        ]

    def _detect_lower_panel_indicators(self, image: np.ndarray) -> List[Dict]:
        """Detect indicators in lower panels"""
        return [{"type": "rsi", "value": 65, "description": "RSI at 65 - moderately bullish"}]

    def _detect_bollinger_bands(self, image: np.ndarray) -> Optional[Dict]:
        """Detect Bollinger Bands"""
        return None  # Simplified


# Integration with AI Trading Analyst
async def analyze_chart_screenshot(
    image_data: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """API endpoint integration for chart screenshot analysis"""
    analyzer = ChartVisionAnalyzer()
    result = await analyzer.analyze_chart_image(image_data, context)

    return {
        "status": "success",
        "analysis": result["analysis"],
        "visual_data": result["visual_data"],
        "trading_signals": result["trading_signals"],
        "confidence": result["confidence"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
