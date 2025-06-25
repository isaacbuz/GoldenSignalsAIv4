"""
Volatility Agent - Comprehensive volatility analysis for options and equity trading.
Analyzes ATR, realized vs implied volatility, volatility skew, vol of vol, and volatility regimes.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.stats import norm
import math
from src.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class VolatilityAgent(BaseAgent):
    """Agent that analyzes volatility patterns and generates volatility-based signals."""
    
    def __init__(
        self,
        name: str = "Volatility",
        lookback_period: int = 30,
        short_vol_period: int = 10,
        long_vol_period: int = 60,
        vol_percentile_period: int = 252,
        vol_spike_threshold: float = 2.0,
        vol_contraction_threshold: float = 0.5,
        skew_threshold: float = 0.15
    ):
        """
        Initialize Volatility agent.
        
        Args:
            name: Agent name
            lookback_period: Default lookback for calculations
            short_vol_period: Short-term volatility period
            long_vol_period: Long-term volatility period  
            vol_percentile_period: Period for volatility percentile ranking
            vol_spike_threshold: Volatility spike threshold (2x average)
            vol_contraction_threshold: Volatility contraction threshold (0.5x average)
            skew_threshold: Significant skew threshold
        """
        super().__init__(name=name, agent_type="volatility")
        self.lookback_period = lookback_period
        self.short_vol_period = short_vol_period
        self.long_vol_period = long_vol_period
        self.vol_percentile_period = vol_percentile_period
        self.vol_spike_threshold = vol_spike_threshold
        self.vol_contraction_threshold = vol_contraction_threshold
        self.skew_threshold = skew_threshold
        
    def calculate_realized_volatility(self, prices: pd.Series, period: int = None) -> Optional[Dict[str, float]]:
        """Calculate realized volatility using various methods."""
        try:
            if period is None:
                period = self.lookback_period
                
            if len(prices) < period + 1:
                return None
            
            # Simple return volatility (close-to-close)
            returns = prices.pct_change().dropna()
            
            if len(returns) < period:
                return None
            
            recent_returns = returns.tail(period)
            
            # Annualized volatility
            daily_vol = recent_returns.std()
            annualized_vol = daily_vol * np.sqrt(252)
            
            # EWMA volatility (gives more weight to recent observations)
            ewma_vol = returns.ewm(span=period).std().iloc[-1] * np.sqrt(252)
            
            # Yang-Zhang volatility (uses OHLC data if available)
            yz_vol = annualized_vol  # Simplified, would need OHLC for true YZ
            
            return {
                'daily_vol': daily_vol,
                'annualized_vol': annualized_vol,
                'ewma_vol': ewma_vol,
                'yang_zhang_vol': yz_vol,
                'vol_of_vol': recent_returns.rolling(window=min(10, len(recent_returns))).std().std()
            }
            
        except Exception as e:
            logger.error(f"Realized volatility calculation failed: {str(e)}")
            return None
    
    def calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Optional[Dict[str, float]]:
        """Calculate Average True Range and related metrics."""
        try:
            if len(highs) < period + 1:
                return None
            
            # True Range calculation
            tr1 = highs - lows
            tr2 = abs(highs - closes.shift(1))
            tr3 = abs(lows - closes.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR (Simple Moving Average of True Range)
            atr = true_range.rolling(window=period).mean()
            current_atr = atr.iloc[-1]
            
            # ATR as percentage of price
            current_price = closes.iloc[-1]
            atr_percent = (current_atr / current_price) * 100
            
            # ATR trend
            atr_slope = None
            if len(atr) >= 5:
                recent_atr = atr.tail(5).dropna()
                if len(recent_atr) >= 3:
                    atr_slope, _, atr_r_value, _, _ = stats.linregress(range(len(recent_atr)), recent_atr)
            
            # ATR percentile
            if len(atr) >= self.vol_percentile_period:
                atr_percentile = stats.percentileofscore(atr.tail(self.vol_percentile_period), current_atr)
            else:
                atr_percentile = stats.percentileofscore(atr.dropna(), current_atr)
            
            return {
                'atr': current_atr,
                'atr_percent': atr_percent,
                'atr_slope': atr_slope,
                'atr_percentile': atr_percentile,
                'high_volatility': atr_percentile > 80,
                'low_volatility': atr_percentile < 20
            }
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {str(e)}")
            return None
    
    def analyze_volatility_regime(self, prices: pd.Series) -> Optional[Dict[str, Any]]:
        """Analyze current volatility regime and trends."""
        try:
            if len(prices) < max(self.short_vol_period, self.long_vol_period) + 1:
                return None
            
            # Calculate short and long-term volatility
            short_vol = self.calculate_realized_volatility(prices, self.short_vol_period)
            long_vol = self.calculate_realized_volatility(prices, self.long_vol_period)
            
            if not short_vol or not long_vol:
                return None
            
            short_term_vol = short_vol['annualized_vol']
            long_term_vol = long_vol['annualized_vol']
            
            # Volatility ratio
            vol_ratio = short_term_vol / long_term_vol if long_term_vol > 0 else 1
            
            # Determine regime
            if vol_ratio > self.vol_spike_threshold:
                regime = 'high_volatility'
                regime_signal = 'volatility_spike'
            elif vol_ratio < self.vol_contraction_threshold:
                regime = 'low_volatility'
                regime_signal = 'volatility_contraction'
            elif vol_ratio > 1.2:
                regime = 'rising_volatility'
                regime_signal = 'volatility_expansion'
            elif vol_ratio < 0.8:
                regime = 'falling_volatility'
                regime_signal = 'volatility_compression'
            else:
                regime = 'normal_volatility'
                regime_signal = 'stable'
            
            # Calculate volatility percentile over longer period
            returns = prices.pct_change().dropna()
            if len(returns) >= self.vol_percentile_period:
                vol_series = returns.rolling(window=self.lookback_period).std() * np.sqrt(252)
                vol_percentile = stats.percentileofscore(vol_series.dropna(), short_term_vol)
            else:
                vol_percentile = 50  # Default to median
            
            # Volatility clustering (GARCH-like effect)
            recent_returns = returns.tail(20)
            vol_clustering = recent_returns.rolling(window=5).std().std()
            
            return {
                'regime': regime,
                'regime_signal': regime_signal,
                'vol_ratio': vol_ratio,
                'short_term_vol': short_term_vol,
                'long_term_vol': long_term_vol,
                'vol_percentile': vol_percentile,
                'vol_clustering': vol_clustering,
                'extreme_vol': vol_percentile > 90 or vol_percentile < 10
            }
            
        except Exception as e:
            logger.error(f"Volatility regime analysis failed: {str(e)}")
            return None
    
    def calculate_volatility_skew(self, prices: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate volatility skew and term structure indicators."""
        try:
            if len(prices) < self.lookback_period + 1:
                return None
            
            returns = prices.pct_change().dropna()
            recent_returns = returns.tail(self.lookback_period)
            
            # Return distribution skewness
            skewness = stats.skew(recent_returns)
            kurtosis = stats.kurtosis(recent_returns)
            
            # Upside vs downside volatility
            positive_returns = recent_returns[recent_returns > 0]
            negative_returns = recent_returns[recent_returns < 0]
            
            upside_vol = positive_returns.std() if len(positive_returns) > 1 else 0
            downside_vol = negative_returns.std() if len(negative_returns) > 1 else 0
            
            # Volatility skew ratio
            vol_skew_ratio = downside_vol / upside_vol if upside_vol > 0 else float('inf')
            
            # Risk-adjusted metrics
            sharpe_ratio = recent_returns.mean() / recent_returns.std() if recent_returns.std() > 0 else 0
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'upside_vol': upside_vol,
                'downside_vol': downside_vol,
                'vol_skew_ratio': vol_skew_ratio,
                'sharpe_ratio': sharpe_ratio,
                'fat_tails': abs(kurtosis) > 3,
                'negative_skew': skewness < -self.skew_threshold,
                'positive_skew': skewness > self.skew_threshold
            }
            
        except Exception as e:
            logger.error(f"Volatility skew calculation failed: {str(e)}")
            return None
    
    def detect_volatility_patterns(self, prices: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect specific volatility patterns and their trading implications."""
        try:
            if len(prices) < self.lookback_period + 1:
                return None
            
            patterns = []
            
            # Get volatility metrics
            vol_regime = self.analyze_volatility_regime(prices)
            if not vol_regime:
                return None
            
            # Pattern 1: Volatility squeeze (low vol before expansion)
            if vol_regime['regime'] == 'low_volatility' and vol_regime['vol_percentile'] < 20:
                patterns.append({
                    'type': 'volatility_squeeze',
                    'signal': 'prepare_for_breakout',
                    'direction': 'neutral',
                    'strength': (20 - vol_regime['vol_percentile']) / 20
                })
            
            # Pattern 2: Volatility spike (mean reversion opportunity)
            elif vol_regime['regime'] == 'high_volatility' and vol_regime['vol_percentile'] > 80:
                patterns.append({
                    'type': 'volatility_spike',
                    'signal': 'vol_mean_reversion',
                    'direction': 'sell_vol',
                    'strength': (vol_regime['vol_percentile'] - 80) / 20
                })
            
            # Pattern 3: Volatility expansion (momentum)
            elif vol_regime['regime_signal'] == 'volatility_expansion':
                patterns.append({
                    'type': 'volatility_expansion',
                    'signal': 'momentum_continuation',
                    'direction': 'trend_follow',
                    'strength': min(vol_regime['vol_ratio'] - 1, 1.0)
                })
            
            # Pattern 4: Volatility compression (range trading)
            elif vol_regime['regime_signal'] == 'volatility_compression':
                patterns.append({
                    'type': 'volatility_compression',
                    'signal': 'range_trading',
                    'direction': 'mean_revert',
                    'strength': min(1 - vol_regime['vol_ratio'], 1.0)
                })
            
            # Check for volatility clustering
            if vol_regime['vol_clustering'] > vol_regime['short_term_vol'] * 0.1:
                patterns.append({
                    'type': 'volatility_clustering',
                    'signal': 'garch_effect',
                    'direction': 'vol_persistence',
                    'strength': 0.6
                })
            
            return {
                'patterns': patterns,
                'pattern_count': len(patterns),
                'dominant_pattern': patterns[0] if patterns else None
            }
            
        except Exception as e:
            logger.error(f"Volatility pattern detection failed: {str(e)}")
            return None
    
    def generate_volatility_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on volatility analysis."""
        try:
            prices = pd.Series(data["close_prices"])
            highs = pd.Series(data.get("high_prices", prices))
            lows = pd.Series(data.get("low_prices", prices))
            
            # Get all volatility metrics
            vol_regime = self.analyze_volatility_regime(prices)
            atr_data = self.calculate_atr(highs, lows, prices)
            vol_skew = self.calculate_volatility_skew(prices)
            vol_patterns = self.detect_volatility_patterns(prices)
            
            if not vol_regime:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'no_signal',
                    'reasoning': 'Insufficient data for volatility analysis'
                }
            
            # Determine primary signal
            action = "hold"
            confidence = 0.0
            signal_type = "neutral"
            reasoning = []
            
            # Volatility regime-based signals
            if vol_regime['regime'] == 'low_volatility' and vol_regime['vol_percentile'] < 25:
                # Low volatility - expect expansion
                action = "buy"  # Buy options (long volatility)
                confidence += 0.4
                signal_type = "long_volatility"
                reasoning.append("Low volatility regime - expect expansion")
                
            elif vol_regime['regime'] == 'high_volatility' and vol_regime['vol_percentile'] > 75:
                # High volatility - expect contraction
                action = "sell"  # Sell options (short volatility)
                confidence += 0.4
                signal_type = "short_volatility"
                reasoning.append("High volatility regime - expect mean reversion")
            
            # Pattern-based signals
            if vol_patterns and vol_patterns['patterns']:
                dominant_pattern = vol_patterns['dominant_pattern']
                
                if dominant_pattern['type'] == 'volatility_squeeze':
                    confidence += 0.3
                    reasoning.append("Volatility squeeze detected - breakout expected")
                    
                elif dominant_pattern['type'] == 'volatility_spike':
                    confidence += 0.3
                    reasoning.append("Volatility spike - mean reversion opportunity")
                    
                elif dominant_pattern['type'] == 'volatility_expansion':
                    if action == "hold":
                        action = "buy"
                    confidence += 0.2
                    reasoning.append("Volatility expansion - momentum signal")
            
            # ATR-based confirmation
            if atr_data:
                if atr_data['high_volatility'] and signal_type == "short_volatility":
                    confidence *= 1.2
                elif atr_data['low_volatility'] and signal_type == "long_volatility":
                    confidence *= 1.2
            
            # Skew-based adjustments
            if vol_skew:
                if vol_skew['negative_skew'] and action == "buy":
                    confidence *= 0.9  # Reduce bullish confidence with negative skew
                    reasoning.append("Negative skew detected")
                elif vol_skew['fat_tails']:
                    reasoning.append("Fat tails detected - higher risk")
            
            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Volatility signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate volatility-based signals."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")
            
            prices = pd.Series(data["close_prices"])
            highs = pd.Series(data.get("high_prices", prices))
            lows = pd.Series(data.get("low_prices", prices))
            
            if len(prices) < max(self.short_vol_period, self.long_vol_period) + 1:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data for volatility analysis"}
                }
            
            # Generate signals
            signal_data = self.generate_volatility_signals(data)
            
            # Get comprehensive volatility metrics
            vol_regime = self.analyze_volatility_regime(prices)
            atr_data = self.calculate_atr(highs, lows, prices)
            vol_skew = self.calculate_volatility_skew(prices)
            vol_patterns = self.detect_volatility_patterns(prices)
            
            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "volatility_regime": vol_regime,
                    "atr_data": atr_data,
                    "volatility_skew": vol_skew,
                    "volatility_patterns": vol_patterns
                }
            }
            
        except Exception as e:
            logger.error(f"Volatility signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 