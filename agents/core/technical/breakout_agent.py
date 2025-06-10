"""
Breakout Agent - Detects breakouts from recent price ranges and key levels.
Identifies when price breaks above resistance or below support with volume confirmation.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from ..base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class BreakoutAgent(BaseAgent):
    """Agent that detects breakouts from consolidation ranges and key levels."""
    
    def __init__(
        self,
        name: str = "Breakout",
        lookback_period: int = 20,
        breakout_threshold: float = 0.02,
        volume_multiplier: float = 1.5,
        min_consolidation_bars: int = 5,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize Breakout agent.
        
        Args:
            name: Agent name
            lookback_period: Period to look back for range calculation
            breakout_threshold: Minimum percentage breakout (2%)
            volume_multiplier: Volume should be X times average
            min_consolidation_bars: Minimum bars for consolidation
            atr_multiplier: ATR multiplier for dynamic threshold
        """
        super().__init__(name=name, agent_type="technical")
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_multiplier = volume_multiplier
        self.min_consolidation_bars = min_consolidation_bars
        self.atr_multiplier = atr_multiplier
        
    def calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range for dynamic threshold setting."""
        try:
            if len(highs) < period + 1:
                return None
                
            # Calculate true range
            tr1 = highs - lows
            tr2 = abs(highs - closes.shift(1))
            tr3 = abs(lows - closes.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr)
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {str(e)}")
            return None
    
    def find_support_resistance(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Tuple[float, float]:
        """Find recent support and resistance levels."""
        try:
            recent_highs = highs.tail(self.lookback_period)
            recent_lows = lows.tail(self.lookback_period)
            recent_closes = closes.tail(self.lookback_period)
            
            # Simple approach: use recent high/low
            resistance = recent_highs.max()
            support = recent_lows.min()
            
            # More sophisticated: find frequently tested levels
            price_levels = pd.concat([recent_highs, recent_lows, recent_closes]).values
            
            # Group similar price levels (within 1% of each other)
            tolerance = 0.01
            level_counts = {}
            
            for price in price_levels:
                found_level = False
                for level in level_counts:
                    if abs(price - level) / level < tolerance:
                        level_counts[level] += 1
                        found_level = True
                        break
                if not found_level:
                    level_counts[price] = 1
            
            # Find most tested levels
            if level_counts:
                sorted_levels = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Update resistance and support based on most tested levels
                current_price = closes.iloc[-1]
                for level, count in sorted_levels:
                    if level > current_price and count >= 2:
                        resistance = min(resistance, level)
                    elif level < current_price and count >= 2:
                        support = max(support, level)
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Support/resistance calculation failed: {str(e)}")
            return lows.tail(self.lookback_period).min(), highs.tail(self.lookback_period).max()
    
    def detect_consolidation(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect if price is in consolidation before breakout."""
        try:
            if len(closes) < self.min_consolidation_bars:
                return None
            
            recent_data = closes.tail(self.lookback_period)
            
            # Calculate range characteristics
            range_high = recent_data.max()
            range_low = recent_data.min()
            range_size = (range_high - range_low) / range_low
            
            # Check for consolidation (low volatility)
            volatility = recent_data.pct_change().std()
            
            # Count how many bars stayed within the range
            bars_in_range = 0
            for price in recent_data:
                if range_low <= price <= range_high:
                    bars_in_range += 1
            
            consolidation_ratio = bars_in_range / len(recent_data)
            
            # Determine if we have consolidation
            is_consolidation = (
                range_size < 0.10 and  # Range < 10%
                volatility < 0.02 and  # Low volatility
                consolidation_ratio > 0.8  # Most bars in range
            )
            
            if is_consolidation:
                return {
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_size': range_size,
                    'volatility': volatility,
                    'consolidation_ratio': consolidation_ratio
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Consolidation detection failed: {str(e)}")
            return None
    
    def detect_breakout(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect breakout from consolidation or key levels."""
        try:
            closes = pd.Series(data["close_prices"])
            highs = pd.Series(data.get("high_prices", closes))
            lows = pd.Series(data.get("low_prices", closes))
            volume = pd.Series(data.get("volume", []))
            
            current_price = closes.iloc[-1]
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]
            
            # Find support and resistance
            support, resistance = self.find_support_resistance(highs, lows, closes)
            
            # Calculate dynamic threshold using ATR
            atr = self.calculate_atr(highs, lows, closes)
            dynamic_threshold = self.breakout_threshold
            if atr is not None:
                dynamic_threshold = max(self.breakout_threshold, (atr * self.atr_multiplier) / current_price)
            
            # Check for breakout
            breakout_type = None
            breakout_level = None
            confidence = 0.0
            
            # Resistance breakout (bullish)
            if current_high > resistance * (1 + dynamic_threshold):
                breakout_type = 'resistance_breakout'
                breakout_level = resistance
                distance = (current_high - resistance) / resistance
                confidence = min(1.0, distance / dynamic_threshold)
            
            # Support breakdown (bearish)
            elif current_low < support * (1 - dynamic_threshold):
                breakout_type = 'support_breakdown'
                breakout_level = support
                distance = (support - current_low) / support
                confidence = min(1.0, distance / dynamic_threshold)
            
            if breakout_type is None:
                return None
            
            # Volume confirmation
            volume_confirmed = True
            if len(volume) > 1:
                current_volume = volume.iloc[-1]
                avg_volume = volume.tail(self.lookback_period).mean()
                
                if current_volume > avg_volume * self.volume_multiplier:
                    confidence *= 1.2  # Boost confidence with volume
                    volume_confirmed = True
                else:
                    confidence *= 0.8  # Reduce confidence without volume
                    volume_confirmed = False
            
            # Check for false breakout (price quickly reverses)
            false_breakout_check = False
            if len(closes) >= 3:
                if breakout_type == 'resistance_breakout':
                    # Check if price closed back below resistance
                    false_breakout_check = current_price < resistance
                elif breakout_type == 'support_breakdown':
                    # Check if price closed back above support
                    false_breakout_check = current_price > support
                    
                if false_breakout_check:
                    confidence *= 0.5  # Significantly reduce confidence
            
            return {
                'breakout_type': breakout_type,
                'breakout_level': breakout_level,
                'current_price': current_price,
                'confidence': min(1.0, max(0.0, confidence)),
                'volume_confirmed': volume_confirmed,
                'dynamic_threshold': dynamic_threshold,
                'atr': atr,
                'false_breakout_risk': false_breakout_check
            }
            
        except Exception as e:
            logger.error(f"Breakout detection failed: {str(e)}")
            return None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and detect breakout signals."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")
            
            closes = pd.Series(data["close_prices"])
            
            if len(closes) < self.lookback_period:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data for breakout detection"}
                }
            
            # Check for consolidation first
            highs = pd.Series(data.get("high_prices", closes))
            lows = pd.Series(data.get("low_prices", closes))
            
            consolidation = self.detect_consolidation(highs, lows, closes)
            
            # Detect breakout
            breakout = self.detect_breakout(data)
            
            if breakout is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "consolidation": consolidation,
                        "awaiting_breakout": consolidation is not None
                    }
                }
            
            # Generate signal based on breakout type
            if breakout['breakout_type'] == 'resistance_breakout':
                action = "buy"
            elif breakout['breakout_type'] == 'support_breakdown':
                action = "sell"
            else:
                action = "hold"
            
            return {
                "action": action,
                "confidence": breakout['confidence'],
                "metadata": {
                    "breakout": breakout,
                    "consolidation": consolidation,
                    "support_level": self.find_support_resistance(highs, lows, closes)[0],
                    "resistance_level": self.find_support_resistance(highs, lows, closes)[1]
                }
            }
            
        except Exception as e:
            logger.error(f"Breakout signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 