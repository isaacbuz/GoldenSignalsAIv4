"""
Hybrid Volume Agent - Independent + Collaborative Volume Analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.common.hybrid_agent_base import HybridAgent
from agents.common.data_bus import SharedDataTypes

logger = logging.getLogger(__name__)

class HybridVolumeAgent(HybridAgent):
    """
    Volume Agent with dual signal generation
    
    Independent: Pure volume spike/trend analysis
    Collaborative: Volume + price patterns, support/resistance, momentum
    """
    
    def __init__(self, data_bus=None, spike_threshold: float = 2.0, lookback_period: int = 20):
        super().__init__("HybridVolumeAgent", data_bus)
        self.spike_threshold = spike_threshold
        self.lookback_period = lookback_period
        
    def analyze_independent(self, symbol: str, data: Any = None) -> Dict[str, Any]:
        """Pure volume analysis without external context"""
        try:
            # Fetch data if not provided
            if data is None:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")
            
            if data.empty or len(data) < self.lookback_period:
                return self._create_signal("HOLD", 0.0, "Insufficient data")
            
            # Volume analysis
            volume_metrics = self._analyze_volume(data)
            
            # Generate signal based on volume alone
            action = "HOLD"
            confidence = 0.0
            reasoning = []
            
            # Volume spike detection
            if volume_metrics['is_spike']:
                spike_type = volume_metrics['spike_type']
                
                if spike_type == 'bullish_spike':
                    action = "BUY"
                    confidence = 0.7 + (volume_metrics['z_score'] - 2) * 0.05
                    reasoning.append(f"Bullish volume spike ({volume_metrics['volume_ratio']:.1f}x avg)")
                    
                elif spike_type == 'bearish_spike':
                    action = "SELL"
                    confidence = 0.7 + (volume_metrics['z_score'] - 2) * 0.05
                    reasoning.append(f"Bearish volume spike ({volume_metrics['volume_ratio']:.1f}x avg)")
                    
                elif spike_type == 'absorption':
                    # For independent analysis, absorption is neutral
                    confidence = 0.4
                    reasoning.append(f"High volume absorption detected")
            
            # Volume trend analysis
            if volume_metrics['volume_trend'] > 0.3:
                if action == "HOLD":
                    # Rising volume without spike suggests accumulation
                    action = "BUY"
                    confidence = 0.45
                    reasoning.append("Volume trend increasing (accumulation)")
                else:
                    confidence += 0.05
                    reasoning.append("Volume trend confirms signal")
                    
            elif volume_metrics['volume_trend'] < -0.3:
                if action == "HOLD":
                    # Declining volume suggests distribution
                    action = "SELL"
                    confidence = 0.4
                    reasoning.append("Volume trend decreasing (distribution)")
            
            # Money flow analysis
            if volume_metrics['money_flow']['accumulation']:
                if action == "BUY":
                    confidence += 0.1
                    reasoning.append("Money flow indicates accumulation")
                elif action == "HOLD":
                    action = "BUY"
                    confidence = 0.5
                    reasoning.append("Positive money flow detected")
            
            # Share insights for other agents
            if self.data_bus:
                self._share_volume_insights(symbol, volume_metrics)
            
            confidence = min(confidence, 0.85)  # Cap independent confidence
            
            return self._create_signal(
                action,
                confidence,
                " | ".join(reasoning) if reasoning else "Normal volume activity",
                volume_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in volume independent analysis: {e}")
            return self._create_signal("HOLD", 0.0, f"Error: {str(e)}")
    
    def analyze_collaborative(self, symbol: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Volume analysis enhanced with market context"""
        try:
            # Start with independent analysis
            base_signal = self.analyze_independent(symbol, data)
            
            # Get context
            pattern_data = context['data'].get(SharedDataTypes.PRICE_PATTERNS, {})
            support_resistance = context['data'].get(SharedDataTypes.SUPPORT_RESISTANCE, {})
            momentum_state = context['data'].get(SharedDataTypes.MOMENTUM_STATE, {})
            market_regime = context['data'].get(SharedDataTypes.MARKET_REGIME, {})
            
            # Enhanced analysis
            action = base_signal['action']
            confidence = base_signal['confidence']
            reasons = [base_signal['metadata']['reasoning']]
            adjustments = []
            
            volume_metrics = base_signal['metadata']['indicators']
            current_price = data['Close'].iloc[-1] if data is not None else 0
            
            # Pattern confirmation
            if pattern_data:
                for agent, patterns in pattern_data.items():
                    pattern_info = patterns['data']
                    
                    if pattern_info.get('bullish_pattern') and action == "BUY":
                        confidence += 0.15
                        adjustments.append(f"Volume confirms {pattern_info.get('pattern_name', 'bullish pattern')}")
                    elif pattern_info.get('bearish_pattern') and action == "SELL":
                        confidence += 0.15
                        adjustments.append(f"Volume confirms {pattern_info.get('pattern_name', 'bearish pattern')}")
                    elif pattern_info.get('reversal_pattern'):
                        if volume_metrics['is_spike']:
                            confidence += 0.1
                            adjustments.append("High volume at reversal pattern")
            
            # Support/Resistance context
            if support_resistance and current_price > 0:
                for agent, sr_data in support_resistance.items():
                    levels = sr_data['data']
                    
                    # Check if volume spike near key levels
                    if volume_metrics['is_spike']:
                        near_support = any(abs(current_price - s) / current_price < 0.01 
                                         for s in levels.get('support_levels', []))
                        near_resistance = any(abs(current_price - r) / current_price < 0.01 
                                            for r in levels.get('resistance_levels', []))
                        
                        if near_support and volume_metrics['spike_type'] == 'bullish_spike':
                            confidence += 0.1
                            adjustments.append("Volume spike at support")
                        elif near_resistance and volume_metrics['spike_type'] == 'bearish_spike':
                            confidence += 0.1
                            adjustments.append("Volume spike at resistance")
            
            # Market regime adjustments
            if market_regime:
                for agent, regime_data in market_regime.items():
                    regime = regime_data['data'].get('regime', '')
                    
                    if 'trending' in regime and volume_metrics['volume_trend'] > 0:
                        confidence += 0.05
                        adjustments.append("Volume supports trending market")
                    elif 'ranging' in regime and volume_metrics['spike_type'] == 'absorption':
                        # In ranging markets, absorption often leads to breakouts
                        if action == "HOLD":
                            action = "BUY" if volume_metrics['money_flow']['accumulation'] else "SELL"
                            confidence = 0.6
                            adjustments.append("Absorption in range - potential breakout")
            
            # Momentum confirmation
            if momentum_state:
                for agent, mom_data in momentum_state.items():
                    momentum = mom_data['data'].get('state', '')
                    
                    if momentum == 'bullish' and action == "BUY":
                        confidence += 0.05
                        adjustments.append("Momentum confirms volume signal")
                    elif momentum == 'bearish' and action == "SELL":
                        confidence += 0.05
                        adjustments.append("Momentum confirms volume signal")
                    elif (momentum == 'bullish' and action == "SELL") or \
                         (momentum == 'bearish' and action == "BUY"):
                        confidence -= 0.1
                        adjustments.append("Momentum contradicts volume signal")
            
            # Special case: Climax volume
            if volume_metrics['z_score'] > 4:
                adjustments.append("Climax volume - potential exhaustion")
                confidence = min(confidence * 0.9, 0.8)  # Reduce confidence on extreme volume
            
            # Cap collaborative confidence
            confidence = min(confidence, 0.95)
            
            # Build enhanced reasoning
            if adjustments:
                reasons.extend(adjustments)
            
            return self._create_signal(
                action,
                confidence,
                " | ".join(reasons),
                {
                    **volume_metrics,
                    'context_enhancements': adjustments,
                    'peers_consulted': len(context['data'])
                }
            )
            
        except Exception as e:
            logger.error(f"Error in volume collaborative analysis: {e}")
            return base_signal
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive volume analysis"""
        # Basic metrics
        volume_sma = data['Volume'].rolling(self.lookback_period).mean()
        volume_std = data['Volume'].rolling(self.lookback_period).std()
        current_volume = data['Volume'].iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        
        # Volume ratios and z-score
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        z_score = (current_volume - avg_volume) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0
        
        # Price change
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        
        # Volume trend
        recent_volume = data['Volume'].tail(5).mean()
        older_volume = data['Volume'].tail(20).head(15).mean()
        volume_trend = (recent_volume - older_volume) / older_volume if older_volume > 0 else 0
        
        # Money flow
        money_flow = self._calculate_money_flow(data)
        
        # Volume profile
        volume_profile = self._calculate_volume_profile(data)
        
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
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow[data['Close'] > data['Open']].tail(10).sum()
        negative_flow = money_flow[data['Close'] <= data['Open']].tail(10).sum()
        
        flow_ratio = positive_flow / negative_flow if negative_flow > 0 else 10
        
        return {
            'positive_flow': float(positive_flow),
            'negative_flow': float(negative_flow),
            'flow_ratio': float(flow_ratio),
            'net_flow': float(positive_flow - negative_flow),
            'accumulation': flow_ratio > 1.5
        }
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile"""
        recent_data = data.tail(20)
        
        # Price levels with highest volume
        price_bins = pd.qcut(recent_data['Close'], q=5, duplicates='drop')
        volume_by_level = recent_data.groupby(price_bins)['Volume'].sum()
        
        poc_level = volume_by_level.idxmax()
        poc_price = (poc_level.left + poc_level.right) / 2
        
        return {
            'poc': float(poc_price),
            'high_volume_zone': [float(poc_level.left), float(poc_level.right)],
            'current_vs_poc': float((data['Close'].iloc[-1] - poc_price) / poc_price)
        }
    
    def _classify_spike(self, volume_ratio: float, price_change: float) -> str:
        """Classify volume spike type"""
        if volume_ratio < self.spike_threshold:
            return "normal"
        
        if price_change > 0.01:
            return "bullish_spike"
        elif price_change < -0.01:
            return "bearish_spike"
        elif abs(price_change) < 0.002:
            return "absorption"
        else:
            return "neutral_spike"
    
    def _share_volume_insights(self, symbol: str, volume_metrics: Dict[str, Any]):
        """Share volume insights via data bus"""
        if not self.data_bus:
            return
        
        # Share volume spike
        if volume_metrics['is_spike']:
            self.data_bus.publish(
                self.name,
                symbol,
                SharedDataTypes.VOLUME_SPIKES,
                {
                    'spike_type': volume_metrics['spike_type'],
                    'volume_ratio': volume_metrics['volume_ratio'],
                    'z_score': volume_metrics['z_score']
                }
            )
        
        # Share volume profile
        self.data_bus.publish(
            self.name,
            symbol,
            SharedDataTypes.VOLUME_PROFILE,
            volume_metrics['volume_profile']
        )
        
        # Share accumulation/distribution
        if volume_metrics['money_flow']['accumulation']:
            self.data_bus.publish(
                self.name,
                symbol,
                SharedDataTypes.ACCUMULATION_DISTRIBUTION,
                {
                    'state': 'accumulation',
                    'flow_ratio': volume_metrics['money_flow']['flow_ratio'],
                    'net_flow': volume_metrics['money_flow']['net_flow']
                }
            )
    
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
        """Specify what context volume agent needs"""
        if self.data_bus:
            return self.data_bus.get_context(symbol, [
                SharedDataTypes.PRICE_PATTERNS,
                SharedDataTypes.SUPPORT_RESISTANCE,
                SharedDataTypes.MOMENTUM_STATE,
                SharedDataTypes.MARKET_REGIME
            ])
        return {} 