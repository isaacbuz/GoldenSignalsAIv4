"""
Enhanced Volume Spike Agent with Data Sharing
Detects unusual volume patterns and shares insights with other agents
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.common.data_bus import EnrichedAgent, SharedDataTypes

logger = logging.getLogger(__name__)

class EnhancedVolumeSpikeAgent(EnrichedAgent):
    """
    Enhanced Volume Spike Agent that:
    - Detects unusual volume patterns
    - Shares volume insights with other agents
    - Uses price action context from other agents
    - Identifies accumulation/distribution
    """

    def __init__(self, data_bus, spike_threshold: float = 2.0, lookback_period: int = 20):
        super().__init__("EnhancedVolumeSpikeAgent", data_bus)
        self.spike_threshold = spike_threshold
        self.lookback_period = lookback_period
        self.subscribed_symbols = set()

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate signal with enriched context from other agents"""
        try:
            # Get shared context from other agents
            context = self.get_shared_context(symbol, [
                SharedDataTypes.PRICE_PATTERNS,
                SharedDataTypes.SUPPORT_RESISTANCE,
                SharedDataTypes.TREND_DIRECTION,
                SharedDataTypes.ORDER_FLOW
            ])

            # Subscribe to updates if not already
            if symbol not in self.subscribed_symbols:
                self._subscribe_to_peer_insights(symbol)
                self.subscribed_symbols.add(symbol)

            # Fetch market data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")

            if data.empty or len(data) < self.lookback_period:
                return self._create_signal("HOLD", 0.0, "Insufficient data")

            # Analyze volume
            volume_analysis = self._analyze_volume(data)

            # Share volume insights
            self._share_volume_insights(symbol, volume_analysis, data)

            # Generate signal with context
            signal = self._generate_contextual_signal(symbol, volume_analysis, context, data)

            return signal

        except Exception as e:
            logger.error(f"Error in Enhanced Volume Spike agent for {symbol}: {e}")
            return self._create_signal("HOLD", 0.0, f"Error: {str(e)}")

    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive volume analysis"""
        # Basic metrics
        volume_sma = data['Volume'].rolling(self.lookback_period).mean()
        volume_std = data['Volume'].rolling(self.lookback_period).std()
        current_volume = data['Volume'].iloc[-1]
        avg_volume = volume_sma.iloc[-1]

        # Volume patterns
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        z_score = (current_volume - avg_volume) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0

        # Price-volume analysis
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]

        # Volume trend
        recent_volume = data['Volume'].tail(5).mean()
        older_volume = data['Volume'].tail(20).head(15).mean()
        volume_trend = (recent_volume - older_volume) / older_volume if older_volume > 0 else 0

        # Accumulation/Distribution
        money_flow = self._calculate_money_flow(data)

        # Volume profile
        volume_profile = self._calculate_simple_volume_profile(data)

        return {
            'current_volume': int(current_volume),
            'avg_volume': int(avg_volume),
            'volume_ratio': float(volume_ratio),
            'z_score': float(z_score),
            'price_change': float(price_change),
            'volume_trend': float(volume_trend),
            'money_flow': money_flow,
            'volume_profile': volume_profile,
            'is_spike': volume_ratio > self.spike_threshold,
            'spike_type': self._classify_spike(volume_ratio, price_change)
        }

    def _calculate_money_flow(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate money flow indicators"""
        # Simple Money Flow
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        # Positive vs Negative flow
        positive_flow = money_flow[data['Close'] > data['Open']].tail(10).sum()
        negative_flow = money_flow[data['Close'] <= data['Open']].tail(10).sum()

        total_flow = positive_flow + negative_flow
        flow_ratio = positive_flow / negative_flow if negative_flow > 0 else 10

        return {
            'positive_flow': float(positive_flow),
            'negative_flow': float(negative_flow),
            'flow_ratio': float(flow_ratio),
            'net_flow': float(positive_flow - negative_flow),
            'accumulation': flow_ratio > 1.5
        }

    def _calculate_simple_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate simplified volume profile"""
        recent_data = data.tail(20)

        # Find price levels with highest volume
        price_bins = pd.qcut(recent_data['Close'], q=5, duplicates='drop')
        volume_by_level = recent_data.groupby(price_bins)['Volume'].sum()

        # Point of Control (highest volume price)
        poc_level = volume_by_level.idxmax()
        poc_price = (poc_level.left + poc_level.right) / 2

        return {
            'poc': float(poc_price),
            'high_volume_zone': [float(poc_level.left), float(poc_level.right)],
            'current_vs_poc': float((data['Close'].iloc[-1] - poc_price) / poc_price)
        }

    def _classify_spike(self, volume_ratio: float, price_change: float) -> str:
        """Classify the type of volume spike"""
        if volume_ratio < self.spike_threshold:
            return "normal"

        if price_change > 0.01:
            return "bullish_spike"
        elif price_change < -0.01:
            return "bearish_spike"
        elif abs(price_change) < 0.002:
            return "absorption"  # High volume, no price movement
        else:
            return "neutral_spike"

    def _share_volume_insights(self, symbol: str, analysis: Dict[str, Any], data: pd.DataFrame):
        """Share volume insights with other agents"""
        # Share volume spike detection
        if analysis['is_spike']:
            self.publish_insight(
                symbol,
                SharedDataTypes.VOLUME_SPIKES,
                {
                    'spike_type': analysis['spike_type'],
                    'volume_ratio': analysis['volume_ratio'],
                    'z_score': analysis['z_score'],
                    'timestamp': datetime.now().isoformat()
                }
            )

        # Share volume profile
        self.publish_insight(
            symbol,
            SharedDataTypes.VOLUME_PROFILE,
            analysis['volume_profile']
        )

        # Share accumulation/distribution
        if analysis['money_flow']['accumulation']:
            self.publish_insight(
                symbol,
                SharedDataTypes.ACCUMULATION_DISTRIBUTION,
                {
                    'state': 'accumulation',
                    'flow_ratio': analysis['money_flow']['flow_ratio'],
                    'net_flow': analysis['money_flow']['net_flow']
                }
            )

    def _generate_contextual_signal(self, symbol: str, volume_analysis: Dict[str, Any],
                                   context: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal using both volume analysis and shared context"""

        action = "HOLD"
        confidence = 0.0
        reasons = []

        # Get insights from other agents
        pattern_data = context['data'].get(SharedDataTypes.PRICE_PATTERNS, {})
        support_resistance = context['data'].get(SharedDataTypes.SUPPORT_RESISTANCE, {})
        trend_direction = context['data'].get(SharedDataTypes.TREND_DIRECTION, {})

        current_price = data['Close'].iloc[-1]

        # Volume spike analysis
        if volume_analysis['is_spike']:
            spike_type = volume_analysis['spike_type']

            if spike_type == "bullish_spike":
                action = "BUY"
                confidence = 0.7
                reasons.append(f"Bullish volume spike ({volume_analysis['volume_ratio']:.1f}x avg)")

                # Boost confidence if near support
                if support_resistance:
                    for agent, sr_data in support_resistance.items():
                        if 'support_levels' in sr_data['data']:
                            nearest_support = min(sr_data['data']['support_levels'],
                                                 key=lambda x: abs(x - current_price))
                            if abs(current_price - nearest_support) / current_price < 0.01:
                                confidence += 0.1
                                reasons.append("Near support level")

            elif spike_type == "bearish_spike":
                action = "SELL"
                confidence = 0.7
                reasons.append(f"Bearish volume spike ({volume_analysis['volume_ratio']:.1f}x avg)")

            elif spike_type == "absorption":
                # Check context for direction
                if trend_direction:
                    for agent, trend_data in trend_direction.items():
                        if trend_data['data'].get('direction') == 'up':
                            action = "SELL"
                            confidence = 0.6
                            reasons.append("Selling absorption at highs")
                        elif trend_data['data'].get('direction') == 'down':
                            action = "BUY"
                            confidence = 0.6
                            reasons.append("Buying absorption at lows")

        # Accumulation/Distribution patterns
        if volume_analysis['money_flow']['accumulation']:
            if action == "HOLD":
                action = "BUY"
                confidence = 0.5
            elif action == "BUY":
                confidence += 0.1
            reasons.append("Accumulation detected")

        # Volume trend confirmation
        if volume_analysis['volume_trend'] > 0.2:
            reasons.append("Increasing volume trend")
            if action in ["BUY", "SELL"]:
                confidence += 0.05

        # Pattern confirmation
        if pattern_data:
            for agent, patterns in pattern_data.items():
                if patterns['data'].get('bullish_pattern'):
                    if action == "BUY":
                        confidence += 0.1
                        reasons.append("Bullish pattern confirmation")
                elif patterns['data'].get('bearish_pattern'):
                    if action == "SELL":
                        confidence += 0.1
                        reasons.append("Bearish pattern confirmation")

        # Cap confidence
        confidence = min(confidence, 0.95)

        # Build comprehensive metadata
        metadata = {
            'agent': self.name,
            'symbol': symbol,
            'reasoning': ' | '.join(reasons) if reasons else "No significant volume patterns",
            'timestamp': datetime.now().isoformat(),
            'volume_analysis': volume_analysis,
            'context_used': {
                'price_patterns': bool(pattern_data),
                'support_resistance': bool(support_resistance),
                'trend_direction': bool(trend_direction)
            },
            'shared_insights': len(context.get('recent_messages', []))
        }

        return {
            "action": action,
            "confidence": float(confidence),
            "metadata": metadata
        }

    def _subscribe_to_peer_insights(self, symbol: str):
        """Subscribe to relevant insights from other agents"""
        # Subscribe to pattern detection
        self.subscribe_to_insights(
            symbol,
            SharedDataTypes.PRICE_PATTERNS,
            lambda msg: logger.info(f"Volume agent received pattern update: {msg['data']}")
        )

        # Subscribe to order flow
        self.subscribe_to_insights(
            symbol,
            SharedDataTypes.ORDER_FLOW,
            lambda msg: logger.info(f"Volume agent received order flow update: {msg['data']}")
        )

    def _create_signal(self, action: str, confidence: float, reason: str) -> Dict[str, Any]:
        """Create standardized signal format"""
        return {
            "action": action,
            "confidence": confidence,
            "metadata": {
                "agent": self.name,
                "reasoning": reason,
                "timestamp": datetime.now().isoformat()
            }
        }


# Example usage
if __name__ == "__main__":
    from agents.common.data_bus import AgentDataBus

    # Create shared data bus
    data_bus = AgentDataBus()

    # Create enhanced volume agent
    volume_agent = EnhancedVolumeSpikeAgent(data_bus)

    # Simulate other agents publishing data
    data_bus.publish(
        "PatternAgent",
        "AAPL",
        SharedDataTypes.PRICE_PATTERNS,
        {
            'bullish_pattern': True,
            'pattern_name': 'Bull Flag',
            'confidence': 0.8
        }
    )

    data_bus.publish(
        "SupportResistanceAgent",
        "AAPL",
        SharedDataTypes.SUPPORT_RESISTANCE,
        {
            'support_levels': [174.0, 172.5, 170.0],
            'resistance_levels': [178.0, 180.0, 182.5]
        }
    )

    # Generate signal with context
    signal = volume_agent.generate_signal("AAPL")
    print(f"Enhanced Volume Signal: {signal}")
