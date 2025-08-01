import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Candlestick pattern types"""

    # Single candle patterns
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    SPINNING_TOP = "spinning_top"
    MARUBOZU = "marubozu"
    LONG_LEGGED_DOJI = "long_legged_doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    GRAVESTONE_DOJI = "gravestone_doji"

    # Two candle patterns
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    TWEEZER_TOP = "tweezer_top"
    TWEEZER_BOTTOM = "tweezer_bottom"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    BULLISH_HARAMI = "bullish_harami"
    BEARISH_HARAMI = "bearish_harami"

    # Three candle patterns
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"

    # Complex patterns
    RISING_THREE_METHODS = "rising_three_methods"
    FALLING_THREE_METHODS = "falling_three_methods"
    MAT_HOLD = "mat_hold"


@dataclass
class CandlestickPattern:
    """Detected candlestick pattern"""

    pattern_type: PatternType
    index: int
    timestamp: datetime
    price: float
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-100
    confidence: float  # 0-100
    description: str
    success_rate: float  # Historical success rate
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = None


class CandlestickPatternService:
    """Advanced candlestick pattern recognition service"""

    # Pattern success rates based on research
    PATTERN_SUCCESS_RATES = {
        PatternType.HAMMER: 0.62,
        PatternType.INVERTED_HAMMER: 0.60,
        PatternType.SHOOTING_STAR: 0.59,
        PatternType.HANGING_MAN: 0.57,
        PatternType.DOJI: 0.55,
        PatternType.BULLISH_ENGULFING: 0.65,
        PatternType.BEARISH_ENGULFING: 0.72,
        PatternType.MORNING_STAR: 0.65,
        PatternType.EVENING_STAR: 0.69,
        PatternType.THREE_WHITE_SOLDIERS: 0.82,
        PatternType.THREE_BLACK_CROWS: 0.78,
        PatternType.PIERCING_LINE: 0.64,
        PatternType.DARK_CLOUD_COVER: 0.66,
        PatternType.RISING_THREE_METHODS: 0.74,
        PatternType.FALLING_THREE_METHODS: 0.72,
        PatternType.MAT_HOLD: 0.70,
        PatternType.BULLISH_HARAMI: 0.61,
        PatternType.BEARISH_HARAMI: 0.63,
        PatternType.TWEEZER_TOP: 0.58,
        PatternType.TWEEZER_BOTTOM: 0.60,
        PatternType.THREE_INSIDE_UP: 0.65,
        PatternType.THREE_INSIDE_DOWN: 0.67,
        PatternType.MARUBOZU: 0.71,
        PatternType.SPINNING_TOP: 0.54,
        PatternType.LONG_LEGGED_DOJI: 0.56,
        PatternType.DRAGONFLY_DOJI: 0.58,
        PatternType.GRAVESTONE_DOJI: 0.57,
    }

    def __init__(self):
        logger.info("Candlestick Pattern Service initialized")

    def detect_all_patterns(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> List[CandlestickPattern]:
        """Detect all candlestick patterns in the data"""
        patterns = []

        # Ensure we have enough data
        if len(df) < 3:
            return patterns

        # Use recent data for pattern detection
        recent_df = df.tail(lookback) if len(df) > lookback else df

        # Detect TA-Lib patterns
        talib_patterns = self._detect_talib_patterns(recent_df)
        patterns.extend(talib_patterns)

        # Detect custom patterns
        custom_patterns = self._detect_custom_patterns(recent_df)
        patterns.extend(custom_patterns)

        # Sort by timestamp
        patterns.sort(key=lambda x: x.timestamp, reverse=True)

        return patterns

    def _detect_talib_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect patterns using TA-Lib"""
        patterns = []

        # TA-Lib pattern functions mapping
        talib_patterns = {
            # Single candle patterns
            "CDL2CROWS": (talib.CDL2CROWS, PatternType.THREE_BLACK_CROWS, "bearish"),
            "CDL3BLACKCROWS": (talib.CDL3BLACKCROWS, PatternType.THREE_BLACK_CROWS, "bearish"),
            "CDL3WHITESOLDIERS": (
                talib.CDL3WHITESOLDIERS,
                PatternType.THREE_WHITE_SOLDIERS,
                "bullish",
            ),
            "CDLDOJI": (talib.CDLDOJI, PatternType.DOJI, "neutral"),
            "CDLDOJISTAR": (talib.CDLDOJISTAR, PatternType.DOJI, "neutral"),
            "CDLDRAGONFLYDOJI": (talib.CDLDRAGONFLYDOJI, PatternType.DRAGONFLY_DOJI, "bullish"),
            "CDLGRAVESTONEDOJI": (talib.CDLGRAVESTONEDOJI, PatternType.GRAVESTONE_DOJI, "bearish"),
            "CDLHAMMER": (talib.CDLHAMMER, PatternType.HAMMER, "bullish"),
            "CDLHANGINGMAN": (talib.CDLHANGINGMAN, PatternType.HANGING_MAN, "bearish"),
            "CDLINVERTEDHAMMER": (talib.CDLINVERTEDHAMMER, PatternType.INVERTED_HAMMER, "bullish"),
            "CDLSHOOTINGSTAR": (talib.CDLSHOOTINGSTAR, PatternType.SHOOTING_STAR, "bearish"),
            "CDLSPINNINGTOP": (talib.CDLSPINNINGTOP, PatternType.SPINNING_TOP, "neutral"),
            # Two candle patterns
            "CDLENGULFING": (talib.CDLENGULFING, None, None),  # Special handling
            "CDLHARAMI": (talib.CDLHARAMI, None, None),  # Special handling
            "CDLPIERCING": (talib.CDLPIERCING, PatternType.PIERCING_LINE, "bullish"),
            "CDLDARKCLOUDCOVER": (talib.CDLDARKCLOUDCOVER, PatternType.DARK_CLOUD_COVER, "bearish"),
            # Three candle patterns
            "CDLMORNINGSTAR": (talib.CDLMORNINGSTAR, PatternType.MORNING_STAR, "bullish"),
            "CDLEVENINGSTAR": (talib.CDLEVENINGSTAR, PatternType.EVENING_STAR, "bearish"),
            "CDL3INSIDE": (talib.CDL3INSIDE, None, None),  # Special handling
            # Complex patterns
            "CDLRISEFALL3METHODS": (talib.CDLRISEFALL3METHODS, None, None),  # Special handling
            "CDLMATHOLD": (talib.CDLMATHOLD, PatternType.MAT_HOLD, "bullish"),
        }

        open_prices = df["open"].values
        high_prices = df["high"].values
        low_prices = df["low"].values
        close_prices = df["close"].values

        for pattern_name, (func, pattern_type, direction) in talib_patterns.items():
            try:
                result = func(open_prices, high_prices, low_prices, close_prices)

                for i, val in enumerate(result):
                    if val != 0:  # Pattern detected
                        # Special handling for patterns that can be bullish or bearish
                        if pattern_name == "CDLENGULFING":
                            pattern_type = (
                                PatternType.BULLISH_ENGULFING
                                if val > 0
                                else PatternType.BEARISH_ENGULFING
                            )
                            direction = "bullish" if val > 0 else "bearish"
                        elif pattern_name == "CDLHARAMI":
                            pattern_type = (
                                PatternType.BULLISH_HARAMI
                                if val > 0
                                else PatternType.BEARISH_HARAMI
                            )
                            direction = "bullish" if val > 0 else "bearish"
                        elif pattern_name == "CDL3INSIDE":
                            pattern_type = (
                                PatternType.THREE_INSIDE_UP
                                if val > 0
                                else PatternType.THREE_INSIDE_DOWN
                            )
                            direction = "bullish" if val > 0 else "bearish"
                        elif pattern_name == "CDLRISEFALL3METHODS":
                            pattern_type = (
                                PatternType.RISING_THREE_METHODS
                                if val > 0
                                else PatternType.FALLING_THREE_METHODS
                            )
                            direction = "bullish" if val > 0 else "bearish"

                        if pattern_type:
                            pattern = self._create_pattern(df, i, pattern_type, direction, abs(val))
                            patterns.append(pattern)

            except Exception as e:
                logger.warning(f"Error detecting {pattern_name}: {e}")

        return patterns

    def _detect_custom_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect custom candlestick patterns not covered by TA-Lib"""
        patterns = []

        # Detect Marubozu
        marubozu_patterns = self._detect_marubozu(df)
        patterns.extend(marubozu_patterns)

        # Detect Tweezer patterns
        tweezer_patterns = self._detect_tweezers(df)
        patterns.extend(tweezer_patterns)

        # Detect Long-legged Doji
        long_legged_doji = self._detect_long_legged_doji(df)
        patterns.extend(long_legged_doji)

        return patterns

    def _detect_marubozu(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Marubozu patterns (no or very small shadows)"""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]
            body = abs(row["close"] - row["open"])
            total_range = row["high"] - row["low"]

            if total_range > 0:
                body_ratio = body / total_range

                # Marubozu has very little shadow (>95% body)
                if body_ratio > 0.95:
                    direction = "bullish" if row["close"] > row["open"] else "bearish"
                    pattern = self._create_pattern(df, i, PatternType.MARUBOZU, direction, 100)
                    patterns.append(pattern)

        return patterns

    def _detect_tweezers(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Tweezer Top and Bottom patterns"""
        patterns = []

        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]

            # Tweezer Top: Two candles with same high
            high_diff = abs(curr["high"] - prev["high"]) / prev["high"]
            if high_diff < 0.001:  # Within 0.1%
                # Check if at resistance
                if self._is_at_resistance(df, i):
                    pattern = self._create_pattern(df, i, PatternType.TWEEZER_TOP, "bearish", 80)
                    patterns.append(pattern)

            # Tweezer Bottom: Two candles with same low
            low_diff = abs(curr["low"] - prev["low"]) / prev["low"]
            if low_diff < 0.001:  # Within 0.1%
                # Check if at support
                if self._is_at_support(df, i):
                    pattern = self._create_pattern(df, i, PatternType.TWEEZER_BOTTOM, "bullish", 80)
                    patterns.append(pattern)

        return patterns

    def _detect_long_legged_doji(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Long-legged Doji patterns"""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]
            body = abs(row["close"] - row["open"])
            total_range = row["high"] - row["low"]
            upper_shadow = row["high"] - max(row["open"], row["close"])
            lower_shadow = min(row["open"], row["close"]) - row["low"]

            if total_range > 0:
                body_ratio = body / total_range

                # Long-legged Doji: Small body with long shadows
                if body_ratio < 0.1 and upper_shadow > body * 3 and lower_shadow > body * 3:
                    pattern = self._create_pattern(
                        df, i, PatternType.LONG_LEGGED_DOJI, "neutral", 70
                    )
                    patterns.append(pattern)

        return patterns

    def _create_pattern(
        self,
        df: pd.DataFrame,
        index: int,
        pattern_type: PatternType,
        direction: str,
        strength: float,
    ) -> CandlestickPattern:
        """Create a CandlestickPattern object"""
        row = df.iloc[index]

        # Calculate price targets
        atr = self._calculate_atr(df, index)
        price = row["close"]

        if direction == "bullish":
            price_target = price + (2 * atr)  # 2 ATR target
            stop_loss = price - atr  # 1 ATR stop
        elif direction == "bearish":
            price_target = price - (2 * atr)
            stop_loss = price + atr
        else:
            price_target = None
            stop_loss = None

        # Get pattern description
        description = self._get_pattern_description(pattern_type)

        # Calculate confidence based on volume and trend
        confidence = self._calculate_pattern_confidence(df, index, pattern_type, direction)

        return CandlestickPattern(
            pattern_type=pattern_type,
            index=index,
            timestamp=df.index[index],
            price=price,
            direction=direction,
            strength=strength,
            confidence=confidence,
            description=description,
            success_rate=self.PATTERN_SUCCESS_RATES.get(pattern_type, 0.5),
            price_target=price_target,
            stop_loss=stop_loss,
            metadata={
                "volume": row["volume"],
                "atr": atr,
                "body_size": abs(row["close"] - row["open"]),
                "upper_shadow": row["high"] - max(row["open"], row["close"]),
                "lower_shadow": min(row["open"], row["close"]) - row["low"],
            },
        )

    def _calculate_atr(self, df: pd.DataFrame, index: int, period: int = 14) -> float:
        """Calculate ATR at given index"""
        if index < period:
            return df["high"].iloc[: index + 1].std()

        # Simple ATR calculation
        tr_values = []
        for i in range(max(0, index - period + 1), index + 1):
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            prev_close = df["close"].iloc[i - 1] if i > 0 else df["close"].iloc[i]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return np.mean(tr_values)

    def _calculate_pattern_confidence(
        self, df: pd.DataFrame, index: int, pattern_type: PatternType, direction: str
    ) -> float:
        """Calculate confidence score for a pattern"""
        confidence = 50.0  # Base confidence

        # Volume confirmation
        if index > 0:
            volume_ratio = df["volume"].iloc[index] / df["volume"].iloc[index - 1]
            if volume_ratio > 1.5:
                confidence += 10
            elif volume_ratio > 1.2:
                confidence += 5

        # Trend alignment
        if index > 20:
            sma20 = df["close"].iloc[index - 19 : index + 1].mean()
            if direction == "bullish" and df["close"].iloc[index] > sma20:
                confidence += 15
            elif direction == "bearish" and df["close"].iloc[index] < sma20:
                confidence += 15

        # Location in range
        if self._is_at_support(df, index) and direction == "bullish":
            confidence += 10
        elif self._is_at_resistance(df, index) and direction == "bearish":
            confidence += 10

        # Pattern-specific adjustments
        if pattern_type in [PatternType.THREE_WHITE_SOLDIERS, PatternType.THREE_BLACK_CROWS]:
            confidence += 10  # Strong patterns
        elif pattern_type in [PatternType.DOJI, PatternType.SPINNING_TOP]:
            confidence -= 10  # Indecision patterns

        return min(100, max(0, confidence))

    def _is_at_support(self, df: pd.DataFrame, index: int, lookback: int = 20) -> bool:
        """Check if price is at support level"""
        if index < lookback:
            return False

        current_low = df["low"].iloc[index]
        recent_lows = df["low"].iloc[index - lookback : index].min()

        return abs(current_low - recent_lows) / recent_lows < 0.01  # Within 1%

    def _is_at_resistance(self, df: pd.DataFrame, index: int, lookback: int = 20) -> bool:
        """Check if price is at resistance level"""
        if index < lookback:
            return False

        current_high = df["high"].iloc[index]
        recent_highs = df["high"].iloc[index - lookback : index].max()

        return abs(current_high - recent_highs) / recent_highs < 0.01  # Within 1%

    def _get_pattern_description(self, pattern_type: PatternType) -> str:
        """Get description for pattern type"""
        descriptions = {
            PatternType.HAMMER: "Bullish reversal pattern with small body at top and long lower shadow",
            PatternType.INVERTED_HAMMER: "Potential bullish reversal with small body at bottom and long upper shadow",
            PatternType.SHOOTING_STAR: "Bearish reversal pattern with small body at bottom and long upper shadow",
            PatternType.HANGING_MAN: "Potential bearish reversal with small body at top and long lower shadow",
            PatternType.DOJI: "Indecision pattern with open and close nearly equal",
            PatternType.BULLISH_ENGULFING: "Strong bullish reversal where current candle engulfs previous bearish candle",
            PatternType.BEARISH_ENGULFING: "Strong bearish reversal where current candle engulfs previous bullish candle",
            PatternType.MORNING_STAR: "Bullish reversal pattern with three candles indicating trend change",
            PatternType.EVENING_STAR: "Bearish reversal pattern with three candles indicating trend change",
            PatternType.THREE_WHITE_SOLDIERS: "Strong bullish continuation with three consecutive rising candles",
            PatternType.THREE_BLACK_CROWS: "Strong bearish continuation with three consecutive falling candles",
            PatternType.PIERCING_LINE: "Bullish reversal where price opens below previous low but closes above midpoint",
            PatternType.DARK_CLOUD_COVER: "Bearish reversal where price opens above previous high but closes below midpoint",
            PatternType.RISING_THREE_METHODS: "Bullish continuation pattern with consolidation",
            PatternType.FALLING_THREE_METHODS: "Bearish continuation pattern with consolidation",
            PatternType.MAT_HOLD: "Bullish continuation pattern similar to rising three methods",
            PatternType.BULLISH_HARAMI: "Potential bullish reversal with small candle within previous bearish candle",
            PatternType.BEARISH_HARAMI: "Potential bearish reversal with small candle within previous bullish candle",
            PatternType.TWEEZER_TOP: "Bearish reversal with two candles having same high at resistance",
            PatternType.TWEEZER_BOTTOM: "Bullish reversal with two candles having same low at support",
            PatternType.THREE_INSIDE_UP: "Bullish reversal confirmed by third candle",
            PatternType.THREE_INSIDE_DOWN: "Bearish reversal confirmed by third candle",
            PatternType.MARUBOZU: "Strong momentum candle with no or minimal shadows",
            PatternType.SPINNING_TOP: "Indecision pattern with small body and equal shadows",
            PatternType.LONG_LEGGED_DOJI: "Strong indecision with very long shadows on both sides",
            PatternType.DRAGONFLY_DOJI: "Potential bullish reversal with long lower shadow and no upper shadow",
            PatternType.GRAVESTONE_DOJI: "Potential bearish reversal with long upper shadow and no lower shadow",
        }

        return descriptions.get(pattern_type, "Candlestick pattern detected")

    def get_pattern_statistics(self, patterns: List[CandlestickPattern]) -> Dict[str, Any]:
        """Get statistics about detected patterns"""
        if not patterns:
            return {
                "total_patterns": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "avg_confidence": 0,
                "most_common": None,
            }

        bullish = [p for p in patterns if p.direction == "bullish"]
        bearish = [p for p in patterns if p.direction == "bearish"]
        neutral = [p for p in patterns if p.direction == "neutral"]

        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern.pattern_type.value] = (
                pattern_counts.get(pattern.pattern_type.value, 0) + 1
            )

        most_common = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None

        return {
            "total_patterns": len(patterns),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "avg_confidence": np.mean([p.confidence for p in patterns]),
            "avg_success_rate": np.mean([p.success_rate for p in patterns]),
            "most_common": most_common,
            "pattern_distribution": pattern_counts,
            "recent_patterns": [
                {
                    "type": p.pattern_type.value,
                    "direction": p.direction,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in patterns[:5]  # Last 5 patterns
            ],
        }

    def visualize_pattern_data(
        self, pattern: CandlestickPattern, df: pd.DataFrame, context_bars: int = 10
    ) -> Dict[str, Any]:
        """Get visualization data for a specific pattern"""
        start_idx = max(0, pattern.index - context_bars)
        end_idx = min(len(df), pattern.index + context_bars + 1)

        context_data = df.iloc[start_idx:end_idx]

        return {
            "pattern_info": {
                "type": pattern.pattern_type.value,
                "direction": pattern.direction,
                "confidence": pattern.confidence,
                "success_rate": pattern.success_rate,
                "description": pattern.description,
            },
            "price_data": {
                "timestamps": context_data.index.tolist(),
                "open": context_data["open"].tolist(),
                "high": context_data["high"].tolist(),
                "low": context_data["low"].tolist(),
                "close": context_data["close"].tolist(),
                "volume": context_data["volume"].tolist(),
            },
            "pattern_candle_index": pattern.index - start_idx,
            "targets": {"price_target": pattern.price_target, "stop_loss": pattern.stop_loss},
            "annotations": self._get_pattern_annotations(pattern, df),
        }

    def _get_pattern_annotations(
        self, pattern: CandlestickPattern, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Get annotations for pattern visualization"""
        annotations = []

        # Pattern label
        annotations.append(
            {
                "type": "label",
                "timestamp": pattern.timestamp,
                "price": pattern.price,
                "text": pattern.pattern_type.value.replace("_", " ").title(),
                "position": "above" if pattern.direction == "bearish" else "below",
            }
        )

        # Price target line
        if pattern.price_target:
            annotations.append(
                {
                    "type": "horizontal_line",
                    "price": pattern.price_target,
                    "color": "green" if pattern.direction == "bullish" else "red",
                    "style": "dashed",
                    "label": "Target",
                }
            )

        # Stop loss line
        if pattern.stop_loss:
            annotations.append(
                {
                    "type": "horizontal_line",
                    "price": pattern.stop_loss,
                    "color": "red",
                    "style": "dotted",
                    "label": "Stop Loss",
                }
            )

        return annotations
