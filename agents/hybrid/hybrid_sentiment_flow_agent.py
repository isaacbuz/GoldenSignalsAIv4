"""
Hybrid Sentiment Flow Agent
Advanced sentiment aggregation from multiple market sources
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

from agents.common.hybrid_agent_base import HybridAgent

class HybridSentimentFlowAgent(HybridAgent):
    """
    Sophisticated sentiment flow agent that:
    - Tracks options flow sentiment
    - Monitors put/call ratios
    - Analyzes volume sentiment
    - Detects institutional positioning
    - Aggregates social sentiment indicators
    """
    
    def __init__(self, data_bus=None):
        super().__init__(data_bus)
        self.name = "HybridSentimentFlowAgent"
        self.base_indicator = "sentiment_flow"
        
        # Sentiment tracking
        self.options_flow_history = deque(maxlen=50)
        self.put_call_history = deque(maxlen=20)
        self.volume_sentiment_history = deque(maxlen=30)
        self.institutional_flow = deque(maxlen=10)
        
        # Thresholds
        self.extreme_bullish_threshold = 0.8
        self.bullish_threshold = 0.6
        self.bearish_threshold = 0.4
        self.extreme_bearish_threshold = 0.2
        
        # Advanced metrics
        self.smart_money_threshold = 1000000  # $1M
        self.unusual_activity_multiplier = 2.0
        
    def analyze_independent(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pure sentiment flow analysis"""
        
        # Extract sentiment indicators
        sentiment_score = self._calculate_sentiment_score(market_data)
        
        # Determine action based on sentiment
        if sentiment_score > self.extreme_bullish_threshold:
            action = "BUY"
            confidence = 0.85
            sentiment = "strong_bullish"
            reason = f"Extreme bullish sentiment flow ({sentiment_score:.2f})"
        elif sentiment_score > self.bullish_threshold:
            action = "BUY"
            confidence = 0.65
            sentiment = "bullish"
            reason = f"Bullish sentiment flow ({sentiment_score:.2f})"
        elif sentiment_score < self.extreme_bearish_threshold:
            action = "SELL"
            confidence = 0.85
            sentiment = "strong_bearish"
            reason = f"Extreme bearish sentiment flow ({sentiment_score:.2f})"
        elif sentiment_score < self.bearish_threshold:
            action = "SELL"
            confidence = 0.65
            sentiment = "bearish"
            reason = f"Bearish sentiment flow ({sentiment_score:.2f})"
        else:
            action = "HOLD"
            confidence = 0.5
            sentiment = "neutral"
            reason = f"Neutral sentiment flow ({sentiment_score:.2f})"
        
        return {
            "action": action,
            "confidence": confidence,
            "sentiment": sentiment,
            "reasoning": reason,
            "metrics": {
                "sentiment_score": sentiment_score,
                "components": self._get_sentiment_components(market_data)
            }
        }
    
    def analyze_collaborative(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sentiment flow with market context"""
        
        # Get base sentiment
        sentiment_score = self._calculate_sentiment_score(market_data)
        
        # Get collaborative data
        price_trend = self._get_shared_data('price_trend_direction', symbol)
        volume_profile = self._get_shared_data('volume_profile', symbol)
        market_regime = self._get_shared_data('market_regime', symbol)
        
        # Adjust sentiment based on market context
        adjusted_score = sentiment_score
        confidence_multiplier = 1.0
        context_notes = []
        
        # Price trend alignment
        if price_trend:
            trend_direction = price_trend.get('direction', 'neutral')
            if trend_direction == 'bullish' and sentiment_score > 0.5:
                adjusted_score *= 1.1
                confidence_multiplier *= 1.15
                context_notes.append("aligned with bullish trend")
            elif trend_direction == 'bearish' and sentiment_score < 0.5:
                adjusted_score *= 0.9
                confidence_multiplier *= 1.15
                context_notes.append("aligned with bearish trend")
            elif (trend_direction == 'bullish' and sentiment_score < 0.4) or \
                 (trend_direction == 'bearish' and sentiment_score > 0.6):
                context_notes.append("diverging from price trend - potential reversal")
                confidence_multiplier *= 0.85
        
        # Volume profile context
        if volume_profile:
            volume_sentiment = self._analyze_volume_sentiment(volume_profile)
            if volume_sentiment == 'accumulation' and sentiment_score > 0.5:
                adjusted_score *= 1.05
                context_notes.append("volume shows accumulation")
            elif volume_sentiment == 'distribution' and sentiment_score < 0.5:
                adjusted_score *= 0.95
                context_notes.append("volume shows distribution")
        
        # Market regime adjustment
        if market_regime:
            regime = market_regime.get('regime', 'neutral')
            if 'volatile' in regime:
                # In volatile markets, sentiment extremes are more meaningful
                if adjusted_score > 0.7 or adjusted_score < 0.3:
                    confidence_multiplier *= 1.2
                    context_notes.append(f"strong sentiment in {regime}")
                else:
                    confidence_multiplier *= 0.8
                    context_notes.append(f"unclear sentiment in {regime}")
        
        # Determine action with context
        adjusted_score = np.clip(adjusted_score, 0, 1)
        
        if adjusted_score > self.extreme_bullish_threshold:
            action = "BUY"
            confidence = 0.9 * confidence_multiplier
            sentiment = "strong_bullish"
        elif adjusted_score > self.bullish_threshold:
            action = "BUY"
            confidence = 0.7 * confidence_multiplier
            sentiment = "bullish"
        elif adjusted_score < self.extreme_bearish_threshold:
            action = "SELL"
            confidence = 0.9 * confidence_multiplier
            sentiment = "strong_bearish"
        elif adjusted_score < self.bearish_threshold:
            action = "SELL"
            confidence = 0.7 * confidence_multiplier
            sentiment = "bearish"
        else:
            action = "HOLD"
            confidence = 0.5 * confidence_multiplier
            sentiment = "neutral"
        
        confidence = np.clip(confidence, 0, 1)
        
        reason = f"Sentiment flow {sentiment} ({adjusted_score:.2f})"
        if context_notes:
            reason += f" - {', '.join(context_notes)}"
        
        return {
            "action": action,
            "confidence": confidence,
            "sentiment": sentiment,
            "reasoning": reason,
            "metrics": {
                "raw_sentiment": sentiment_score,
                "adjusted_sentiment": adjusted_score,
                "context": context_notes
            }
        }
    
    def _calculate_sentiment_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate comprehensive sentiment score"""
        
        components = self._get_sentiment_components(market_data)
        
        # Weighted average of components
        weights = {
            'options_flow': 0.35,
            'put_call_ratio': 0.25,
            'volume_sentiment': 0.20,
            'institutional': 0.15,
            'momentum': 0.05
        }
        
        sentiment_score = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in components:
                sentiment_score += components[component] * weight
                total_weight += weight
        
        if total_weight > 0:
            sentiment_score /= total_weight
        else:
            sentiment_score = 0.5  # Neutral default
        
        return sentiment_score
    
    def _get_sentiment_components(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract individual sentiment components"""
        
        components = {}
        
        # Options flow sentiment (0-1 scale, 1 = very bullish)
        options_data = market_data.get('options', {})
        if options_data:
            call_volume = options_data.get('call_volume', 0)
            put_volume = options_data.get('put_volume', 0)
            total_volume = call_volume + put_volume
            
            if total_volume > 0:
                # More calls = higher score
                components['options_flow'] = call_volume / total_volume
                
                # Put/Call ratio (inverted for consistent scale)
                pc_ratio = put_volume / call_volume if call_volume > 0 else 2.0
                components['put_call_ratio'] = 1 / (1 + pc_ratio)  # Convert to 0-1 scale
            else:
                components['options_flow'] = 0.5
                components['put_call_ratio'] = 0.5
        
        # Volume sentiment
        volume_data = market_data.get('volume', {})
        if volume_data:
            buy_volume = volume_data.get('buy_volume', 0)
            sell_volume = volume_data.get('sell_volume', 0)
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                components['volume_sentiment'] = buy_volume / total_volume
            else:
                components['volume_sentiment'] = 0.5
        
        # Institutional flow (smart money)
        if options_data:
            large_trades = options_data.get('large_trades', [])
            if large_trades:
                bullish_institutional = sum(1 for trade in large_trades 
                                          if trade.get('type') == 'call' and 
                                          trade.get('premium', 0) > self.smart_money_threshold)
                bearish_institutional = sum(1 for trade in large_trades 
                                          if trade.get('type') == 'put' and 
                                          trade.get('premium', 0) > self.smart_money_threshold)
                
                total_institutional = bullish_institutional + bearish_institutional
                if total_institutional > 0:
                    components['institutional'] = bullish_institutional / total_institutional
                else:
                    components['institutional'] = 0.5
        
        # Price momentum component
        price_data = market_data.get('price', {})
        if price_data:
            current = price_data.get('current', 0)
            open_price = price_data.get('open', current)
            
            if open_price > 0:
                momentum = (current - open_price) / open_price
                # Convert to 0-1 scale with sigmoid
                components['momentum'] = 1 / (1 + np.exp(-momentum * 100))
            else:
                components['momentum'] = 0.5
        
        return components
    
    def _analyze_volume_sentiment(self, volume_profile: Dict[str, Any]) -> str:
        """Analyze volume profile for accumulation/distribution"""
        
        profile = volume_profile.get('profile', {})
        poc = volume_profile.get('poc', 0)  # Point of Control
        
        if not profile:
            return 'neutral'
        
        # Simple analysis: compare volume above vs below POC
        above_poc_volume = sum(v for p, v in profile.items() if float(p) > poc)
        below_poc_volume = sum(v for p, v in profile.items() if float(p) <= poc)
        
        total_volume = above_poc_volume + below_poc_volume
        if total_volume == 0:
            return 'neutral'
        
        above_ratio = above_poc_volume / total_volume
        
        if above_ratio > 0.6:
            return 'accumulation'
        elif above_ratio < 0.4:
            return 'distribution'
        else:
            return 'neutral'
    
    def _publish_sentiment_data(self, symbol: str, sentiment_score: float, components: Dict[str, float]):
        """Publish sentiment data for other agents"""
        if self.data_bus:
            self.data_bus.publish('sentiment_score', {
                'symbol': symbol,
                'score': sentiment_score,
                'components': components,
                'timestamp': datetime.now()
            })
            
            # Also publish specific sentiment indicators
            if 'put_call_ratio' in components:
                pc_ratio = 1 / components['put_call_ratio'] - 1 if components['put_call_ratio'] > 0 else 1.0
                self.data_bus.publish('sentiment_put_call_ratio', {
                    'symbol': symbol,
                    'ratio': pc_ratio,
                    'trend': 'bullish' if pc_ratio < 0.7 else 'bearish' if pc_ratio > 1.3 else 'neutral',
                    'timestamp': datetime.now()
                })
            
            if 'options_flow' in components:
                bias = 'bullish' if components['options_flow'] > 0.6 else 'bearish' if components['options_flow'] < 0.4 else 'neutral'
                self.data_bus.publish('sentiment_options_flow', {
                    'symbol': symbol,
                    'bias': bias,
                    'magnitude': abs(components['options_flow'] - 0.5) * 2,
                    'timestamp': datetime.now()
                })
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal with sentiment publishing"""
        
        # Calculate sentiment components
        components = self._get_sentiment_components(market_data)
        sentiment_score = self._calculate_sentiment_score(market_data)
        
        # Publish sentiment data
        self._publish_sentiment_data(symbol, sentiment_score, components)
        
        # Generate signal using base class method
        signal = super().generate_signal(symbol, market_data)
        
        # Add sentiment flow specific metadata
        signal['metadata']['sentiment_flow'] = {
            'overall_score': sentiment_score,
            'components': components,
            'unusual_activity': self._detect_unusual_activity(market_data),
            'smart_money_direction': self._get_smart_money_direction(market_data)
        }
        
        return signal
    
    def _detect_unusual_activity(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unusual options or volume activity"""
        
        unusual = {
            'detected': False,
            'type': None,
            'magnitude': 0
        }
        
        # Check options volume
        options_data = market_data.get('options', {})
        if options_data:
            current_volume = options_data.get('volume', 0)
            avg_volume = options_data.get('avg_volume', 1)
            
            if avg_volume > 0 and current_volume > avg_volume * self.unusual_activity_multiplier:
                unusual['detected'] = True
                unusual['type'] = 'options_volume_spike'
                unusual['magnitude'] = current_volume / avg_volume
        
        # Check for unusual put/call ratios
        if not unusual['detected'] and options_data:
            pc_ratio = options_data.get('put_call_ratio', 1.0)
            if pc_ratio < 0.5 or pc_ratio > 2.0:
                unusual['detected'] = True
                unusual['type'] = 'extreme_put_call_ratio'
                unusual['magnitude'] = abs(pc_ratio - 1.0)
        
        return unusual
    
    def _get_smart_money_direction(self, market_data: Dict[str, Any]) -> str:
        """Determine smart money direction from large trades"""
        
        options_data = market_data.get('options', {})
        large_trades = options_data.get('large_trades', [])
        
        if not large_trades:
            return 'neutral'
        
        # Sum up smart money flow
        bullish_flow = sum(trade.get('premium', 0) for trade in large_trades 
                          if trade.get('type') == 'call' and 
                          trade.get('premium', 0) > self.smart_money_threshold)
        
        bearish_flow = sum(trade.get('premium', 0) for trade in large_trades 
                          if trade.get('type') == 'put' and 
                          trade.get('premium', 0) > self.smart_money_threshold)
        
        total_flow = bullish_flow + bearish_flow
        
        if total_flow == 0:
            return 'neutral'
        
        bullish_ratio = bullish_flow / total_flow
        
        if bullish_ratio > 0.7:
            return 'strong_bullish'
        elif bullish_ratio > 0.6:
            return 'bullish'
        elif bullish_ratio < 0.3:
            return 'strong_bearish'
        elif bullish_ratio < 0.4:
            return 'bearish'
        else:
            return 'neutral' 