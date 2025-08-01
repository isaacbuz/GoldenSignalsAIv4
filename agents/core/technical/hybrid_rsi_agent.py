"""
Hybrid RSI Agent - Example of Independent + Collaborative Analysis
Shows how agents can maintain their own sentiment while collaborating
"""

import logging
import os
import sys
from typing import Any, Dict

import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.common.data_bus import SharedDataTypes
from agents.common.hybrid_agent_base import HybridAgent

logger = logging.getLogger(__name__)

class HybridRSIAgent(HybridAgent):
    """
    RSI Agent that generates both independent and collaborative signals

    Independent: Pure RSI analysis
    Collaborative: RSI + volume/pattern/support-resistance context
    """

    def __init__(self, data_bus=None, period: int = 14):
        super().__init__("HybridRSIAgent", data_bus)
        self.period = period
        self.oversold_threshold = 30
        self.overbought_threshold = 70

    def analyze_independent(self, symbol: str, data: Any = None) -> Dict[str, Any]:
        """Pure RSI analysis without any external context"""
        try:
            # Fetch data if not provided
            if data is None:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")

            if len(data) < self.period + 1:
                return self._create_signal("HOLD", 0.0, "Insufficient data")

            # Calculate RSI
            rsi_data = self._calculate_rsi(data)
            current_rsi = rsi_data['rsi'].iloc[-1]
            rsi_trend = self._get_rsi_trend(rsi_data['rsi'])

            # Pure RSI logic
            action = "HOLD"
            confidence = 0.0
            reasoning = []

            if current_rsi < self.oversold_threshold:
                action = "BUY"
                confidence = 0.6 + ((self.oversold_threshold - current_rsi) / self.oversold_threshold) * 0.3
                reasoning.append(f"RSI oversold at {current_rsi:.1f}")

                if rsi_trend == "rising":
                    confidence += 0.1
                    reasoning.append("RSI momentum turning up")

            elif current_rsi > self.overbought_threshold:
                action = "SELL"
                confidence = 0.6 + ((current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold)) * 0.3
                reasoning.append(f"RSI overbought at {current_rsi:.1f}")

                if rsi_trend == "falling":
                    confidence += 0.1
                    reasoning.append("RSI momentum turning down")
            else:
                # Check for divergences
                divergence = self._check_divergence(data, rsi_data['rsi'])
                if divergence['type'] == 'bullish':
                    action = "BUY"
                    confidence = 0.5
                    reasoning.append("Bullish RSI divergence")
                elif divergence['type'] == 'bearish':
                    action = "SELL"
                    confidence = 0.5
                    reasoning.append("Bearish RSI divergence")
                else:
                    reasoning.append(f"RSI neutral at {current_rsi:.1f}")

            return self._create_signal(
                action,
                confidence,
                " | ".join(reasoning),
                {
                    'rsi': float(current_rsi),
                    'rsi_trend': rsi_trend,
                    'divergence': divergence
                }
            )

        except Exception as e:
            logger.error(f"Error in RSI independent analysis: {e}")
            return self._create_signal("HOLD", 0.0, f"Error: {str(e)}")

    def analyze_collaborative(self, symbol: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """RSI analysis enhanced with context from other agents"""
        try:
            # Start with independent analysis
            base_signal = self.analyze_independent(symbol, data)

            # Get context data
            volume_data = context['data'].get(SharedDataTypes.VOLUME_SPIKES, {})
            pattern_data = context['data'].get(SharedDataTypes.PRICE_PATTERNS, {})
            support_resistance = context['data'].get(SharedDataTypes.SUPPORT_RESISTANCE, {})
            order_flow = context['data'].get(SharedDataTypes.ORDER_FLOW, {})

            # Enhanced analysis with context
            action = base_signal['action']
            confidence = base_signal['confidence']
            reasons = [base_signal['metadata']['reasoning']]
            adjustments = []

            current_rsi = base_signal['metadata']['indicators']['rsi']

            # Volume confirmation
            if volume_data:
                for agent, vol_info in volume_data.items():
                    if 'spike_type' in vol_info['data']:
                        spike_type = vol_info['data']['spike_type']

                        if action == "BUY" and spike_type == 'bullish_spike':
                            confidence += 0.15
                            adjustments.append("Volume confirms bullish signal")
                        elif action == "SELL" and spike_type == 'bearish_spike':
                            confidence += 0.15
                            adjustments.append("Volume confirms bearish signal")
                        elif (action == "BUY" and spike_type == 'bearish_spike') or \
                             (action == "SELL" and spike_type == 'bullish_spike'):
                            confidence -= 0.1
                            adjustments.append("Volume contradicts signal")

            # Pattern confirmation
            if pattern_data:
                for agent, patterns in pattern_data.items():
                    if patterns['data'].get('bullish_pattern') and action == "BUY":
                        confidence += 0.1
                        adjustments.append(f"Bullish pattern: {patterns['data'].get('pattern_name', 'detected')}")
                    elif patterns['data'].get('bearish_pattern') and action == "SELL":
                        confidence += 0.1
                        adjustments.append(f"Bearish pattern: {patterns['data'].get('pattern_name', 'detected')}")

            # Support/Resistance context
            if support_resistance and data is not None:
                current_price = data['Close'].iloc[-1]

                for agent, sr_data in support_resistance.items():
                    if 'support_levels' in sr_data['data'] and action == "BUY":
                        supports = sr_data['data']['support_levels']
                        nearest_support = min(supports, key=lambda x: abs(x - current_price))

                        if abs(current_price - nearest_support) / current_price < 0.02:
                            confidence += 0.1
                            adjustments.append("Near support level")

                    if 'resistance_levels' in sr_data['data'] and action == "SELL":
                        resistances = sr_data['data']['resistance_levels']
                        nearest_resistance = min(resistances, key=lambda x: abs(x - current_price))

                        if abs(current_price - nearest_resistance) / current_price < 0.02:
                            confidence += 0.1
                            adjustments.append("Near resistance level")

            # Order flow context
            if order_flow:
                for agent, flow_data in order_flow.items():
                    if 'imbalance' in flow_data['data']:
                        imbalance = flow_data['data']['imbalance']

                        if action == "BUY" and imbalance > 0:
                            confidence += 0.05
                            adjustments.append("Positive order flow")
                        elif action == "SELL" and imbalance < 0:
                            confidence += 0.05
                            adjustments.append("Negative order flow")

            # Adjust for extreme RSI with no confirmation
            if current_rsi < 20 and len(adjustments) == 0:
                confidence = min(confidence * 0.8, 0.9)  # Reduce confidence without confirmation
                adjustments.append("Extreme oversold - caution advised")
            elif current_rsi > 80 and len(adjustments) == 0:
                confidence = min(confidence * 0.8, 0.9)
                adjustments.append("Extreme overbought - caution advised")

            # Cap confidence
            confidence = min(confidence, 0.95)

            # Build enhanced reasoning
            if adjustments:
                reasons.extend(adjustments)

            return self._create_signal(
                action,
                confidence,
                " | ".join(reasons),
                {
                    'rsi': float(current_rsi),
                    'context_enhancements': adjustments,
                    'peers_consulted': len(context['data'])
                }
            )

        except Exception as e:
            logger.error(f"Error in RSI collaborative analysis: {e}")
            # Fall back to independent signal
            return base_signal

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return pd.DataFrame({
            'rsi': rsi,
            'gain': gain,
            'loss': loss
        })

    def _get_rsi_trend(self, rsi_series: pd.Series) -> str:
        """Determine RSI trend direction"""
        if len(rsi_series) < 5:
            return "neutral"

        recent_rsi = rsi_series.tail(5)
        trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]

        if trend > 5:
            return "rising"
        elif trend < -5:
            return "falling"
        else:
            return "neutral"

    def _check_divergence(self, price_data: pd.DataFrame, rsi_series: pd.Series) -> Dict[str, Any]:
        """Check for RSI divergences"""
        if len(price_data) < 20:
            return {'type': 'none', 'strength': 0}

        # Find recent highs and lows
        recent_prices = price_data['Close'].tail(20)
        recent_rsi = rsi_series.tail(20)

        # Simple divergence check
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]

        if price_trend > 0.02 and rsi_trend < -5:
            return {'type': 'bearish', 'strength': 0.7}
        elif price_trend < -0.02 and rsi_trend > 5:
            return {'type': 'bullish', 'strength': 0.7}
        else:
            return {'type': 'none', 'strength': 0}

    def _create_signal(self, action: str, confidence: float, reasoning: str,
                      indicators: Dict = None) -> Dict[str, Any]:
        """Create standardized signal"""
        return {
            'action': action,
            'confidence': confidence,
            'metadata': {
                'agent': self.name,
                'reasoning': reasoning,
                'indicators': indicators or {}
            }
        }

    def _get_relevant_context(self, symbol: str) -> Dict[str, Any]:
        """Specify what context RSI agent needs"""
        if self.data_bus:
            return self.data_bus.get_context(symbol, [
                SharedDataTypes.VOLUME_SPIKES,
                SharedDataTypes.PRICE_PATTERNS,
                SharedDataTypes.SUPPORT_RESISTANCE,
                SharedDataTypes.ORDER_FLOW
            ])
        return {}

    def _share_insights(self, symbol: str, signal: Dict[str, Any]):
        """Share RSI insights with other agents"""
        if self.data_bus and 'indicators' in signal['metadata']:
            rsi = signal['metadata']['indicators'].get('rsi')
            if rsi is not None:
                # Share overbought/oversold state
                if rsi < 30:
                    state = 'oversold'
                elif rsi > 70:
                    state = 'overbought'
                else:
                    state = 'neutral'

                self.data_bus.publish(
                    self.name,
                    symbol,
                    SharedDataTypes.OVERBOUGHT_OVERSOLD,
                    {
                        'rsi': rsi,
                        'state': state,
                        'divergence': signal['metadata']['indicators'].get('divergence', {})
                    }
                )


# Example usage
if __name__ == "__main__":
    from agents.common.data_bus import AgentDataBus

    # Create data bus
    data_bus = AgentDataBus()

    # Simulate other agents publishing data
    data_bus.publish("VolumeAgent", "AAPL", SharedDataTypes.VOLUME_SPIKES, {
        'spike_type': 'bullish_spike',
        'volume_ratio': 2.5
    })

    # Create hybrid RSI agent
    rsi_agent = HybridRSIAgent(data_bus)

    # Generate signal (will create both independent and collaborative)
    signal = rsi_agent.generate_signal("AAPL")

    print("\n=== Hybrid RSI Signal ===")
    print(f"Final Action: {signal['action']}")
    print(f"Final Confidence: {signal['confidence']:.2%}")
    print(f"\nIndependent: {signal['metadata']['signal_components']['independent']}")
    print(f"\nCollaborative: {signal['metadata']['signal_components']['collaborative']}")
    print(f"\nWeights: {signal['metadata']['signal_components']['weights']}")

    # Check performance metrics
    print(f"\nPerformance: {rsi_agent.get_performance_metrics()}")
