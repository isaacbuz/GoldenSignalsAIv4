"""
IV Rank Agent - Analyzes implied volatility percentile and rank for options trading signals.
Tracks IV rank, percentile, and mean reversion opportunities in volatility.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats
from ...common.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class IVRankAgent(BaseAgent):
    """Agent that analyzes implied volatility rank and percentile for trading signals."""
    
    def __init__(
        self,
        name: str = "IVRank", 
        iv_rank_period: int = 252,  # 1 year lookback
        high_iv_rank_threshold: float = 80.0,
        low_iv_rank_threshold: float = 20.0,
        extreme_iv_threshold: float = 95.0,
        hv_iv_divergence_threshold: float = 0.15,
        vol_mean_reversion_window: int = 30
    ):
        """
        Initialize IV Rank agent.
        
        Args:
            name: Agent name
            iv_rank_period: Period for IV rank calculation (trading days)
            high_iv_rank_threshold: High IV rank threshold (80th percentile)
            low_iv_rank_threshold: Low IV rank threshold (20th percentile) 
            extreme_iv_threshold: Extreme IV threshold (95th percentile)
            hv_iv_divergence_threshold: HV-IV divergence threshold
            vol_mean_reversion_window: Window for volatility mean reversion analysis
        """
        super().__init__(name=name, agent_type="volatility")
        self.iv_rank_period = iv_rank_period
        self.high_iv_rank_threshold = high_iv_rank_threshold
        self.low_iv_rank_threshold = low_iv_rank_threshold
        self.extreme_iv_threshold = extreme_iv_threshold
        self.hv_iv_divergence_threshold = hv_iv_divergence_threshold
        self.vol_mean_reversion_window = vol_mean_reversion_window
        
    def calculate_iv_rank(self, current_iv: float, historical_iv: List[float]) -> Dict[str, float]:
        """Calculate IV rank and percentile."""
        try:
            if not historical_iv or len(historical_iv) < 30:
                return {'iv_rank': 50.0, 'iv_percentile': 50.0, 'sample_size': len(historical_iv)}
            
            # Use recent history up to specified period
            recent_iv = historical_iv[-self.iv_rank_period:] if len(historical_iv) >= self.iv_rank_period else historical_iv
            
            # Add current IV to the dataset
            iv_dataset = recent_iv + [current_iv]
            iv_array = np.array(iv_dataset)
            
            # Calculate percentile rank of current IV
            iv_percentile = stats.percentileofscore(iv_array, current_iv)
            
            # IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
            min_iv = np.min(iv_array)
            max_iv = np.max(iv_array)
            
            if max_iv != min_iv:
                iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
            else:
                iv_rank = 50.0
            
            return {
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'min_iv': min_iv,
                'max_iv': max_iv,
                'mean_iv': np.mean(iv_array),
                'median_iv': np.median(iv_array),
                'sample_size': len(recent_iv)
            }
            
        except Exception as e:
            logger.error(f"IV rank calculation failed: {str(e)}")
            return {'iv_rank': 50.0, 'iv_percentile': 50.0, 'sample_size': 0}
    
    def analyze_hv_iv_relationship(self, current_iv: float, current_hv: float, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze relationship between Historical Volatility (HV) and Implied Volatility (IV)."""
        try:
            if not historical_data or len(historical_data) < 20:
                return {'hv_iv_spread': 0.0, 'hv_iv_ratio': 1.0, 'divergence_detected': False}
            
            # Extract HV and IV from historical data
            hv_values = [d.get('historical_vol', 0) for d in historical_data if d.get('historical_vol', 0) > 0]
            iv_values = [d.get('implied_vol', 0) for d in historical_data if d.get('implied_vol', 0) > 0]
            
            if len(hv_values) < 10 or len(iv_values) < 10:
                return {'hv_iv_spread': 0.0, 'hv_iv_ratio': 1.0, 'divergence_detected': False}
            
            # Current relationship
            hv_iv_spread = current_iv - current_hv
            hv_iv_ratio = current_iv / current_hv if current_hv > 0 else 1.0
            
            # Historical relationship statistics
            historical_spreads = []
            historical_ratios = []
            
            for i in range(min(len(hv_values), len(iv_values))):
                if hv_values[i] > 0 and iv_values[i] > 0:
                    spread = iv_values[i] - hv_values[i]
                    ratio = iv_values[i] / hv_values[i]
                    historical_spreads.append(spread)
                    historical_ratios.append(ratio)
            
            if not historical_spreads:
                return {'hv_iv_spread': hv_iv_spread, 'hv_iv_ratio': hv_iv_ratio, 'divergence_detected': False}
            
            # Statistical analysis
            mean_spread = np.mean(historical_spreads)
            std_spread = np.std(historical_spreads)
            mean_ratio = np.mean(historical_ratios)
            std_ratio = np.std(historical_ratios)
            
            # Detect divergence (current values significantly different from historical)
            spread_z_score = (hv_iv_spread - mean_spread) / std_spread if std_spread > 0 else 0
            ratio_z_score = (hv_iv_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
            
            # Divergence detection
            divergence_detected = (
                abs(spread_z_score) > 2.0 or  # 2 standard deviations
                abs(ratio_z_score) > 2.0 or
                abs(hv_iv_spread) > self.hv_iv_divergence_threshold
            )
            
            # Determine divergence type
            if divergence_detected:
                if hv_iv_spread > mean_spread + 2 * std_spread:
                    divergence_type = 'iv_premium'  # IV unusually high vs HV
                elif hv_iv_spread < mean_spread - 2 * std_spread:
                    divergence_type = 'iv_discount'  # IV unusually low vs HV
                elif hv_iv_ratio > mean_ratio + 2 * std_ratio:
                    divergence_type = 'iv_expensive'
                elif hv_iv_ratio < mean_ratio - 2 * std_ratio:
                    divergence_type = 'iv_cheap'
                else:
                    divergence_type = 'statistical_divergence'
            else:
                divergence_type = 'normal'
            
            return {
                'hv_iv_spread': hv_iv_spread,
                'hv_iv_ratio': hv_iv_ratio,
                'divergence_detected': divergence_detected,
                'divergence_type': divergence_type,
                'spread_z_score': spread_z_score,
                'ratio_z_score': ratio_z_score,
                'historical_mean_spread': mean_spread,
                'historical_mean_ratio': mean_ratio
            }
            
        except Exception as e:
            logger.error(f"HV-IV relationship analysis failed: {str(e)}")
            return {'hv_iv_spread': 0.0, 'hv_iv_ratio': 1.0, 'divergence_detected': False}
    
    def detect_volatility_mean_reversion_signals(self, iv_rank_data: Dict[str, float], historical_data: List[Dict]) -> Dict[str, Any]:
        """Detect mean reversion opportunities in volatility."""
        try:
            current_rank = iv_rank_data.get('iv_rank', 50)
            current_percentile = iv_rank_data.get('iv_percentile', 50)
            
            # Mean reversion signals based on extreme IV rank
            signals = []
            
            # Extreme high IV rank (mean reversion down expected)
            if current_rank >= self.extreme_iv_threshold:
                signals.append({
                    'type': 'extreme_high_iv',
                    'direction': 'sell_volatility',
                    'strength': 'very_strong',
                    'confidence': 0.8,
                    'reasoning': f"IV rank at {current_rank:.1f}% - extreme high, expect mean reversion"
                })
            elif current_rank >= self.high_iv_rank_threshold:
                signals.append({
                    'type': 'high_iv',
                    'direction': 'sell_volatility',
                    'strength': 'strong',
                    'confidence': 0.6,
                    'reasoning': f"IV rank at {current_rank:.1f}% - high, likely to revert lower"
                })
            
            # Extreme low IV rank (mean reversion up expected)
            elif current_rank <= (100 - self.extreme_iv_threshold):  # 5th percentile
                signals.append({
                    'type': 'extreme_low_iv',
                    'direction': 'buy_volatility',
                    'strength': 'very_strong',
                    'confidence': 0.8,
                    'reasoning': f"IV rank at {current_rank:.1f}% - extreme low, expect mean reversion"
                })
            elif current_rank <= self.low_iv_rank_threshold:
                signals.append({
                    'type': 'low_iv',
                    'direction': 'buy_volatility',
                    'strength': 'strong',
                    'confidence': 0.6,
                    'reasoning': f"IV rank at {current_rank:.1f}% - low, likely to revert higher"
                })
            
            # Volatility clustering analysis
            if len(historical_data) >= self.vol_mean_reversion_window:
                recent_iv_data = historical_data[-self.vol_mean_reversion_window:]
                iv_changes = []
                
                for i in range(1, len(recent_iv_data)):
                    prev_iv = recent_iv_data[i-1].get('implied_vol', 0)
                    curr_iv = recent_iv_data[i].get('implied_vol', 0)
                    if prev_iv > 0 and curr_iv > 0:
                        iv_change = (curr_iv - prev_iv) / prev_iv
                        iv_changes.append(iv_change)
                
                if iv_changes:
                    # Check for volatility clustering (high volatility followed by high volatility)
                    recent_volatility = np.std(iv_changes)
                    historical_volatility = np.std([d.get('implied_vol', 0) for d in historical_data])
                    
                    if recent_volatility > historical_volatility * 1.5:
                        signals.append({
                            'type': 'volatility_clustering',
                            'direction': 'sell_volatility',
                            'strength': 'medium',
                            'confidence': 0.4,
                            'reasoning': 'High volatility clustering detected - expect normalization'
                        })
            
            return {
                'signals': signals,
                'signal_count': len(signals),
                'primary_signal': signals[0] if signals else None
            }
            
        except Exception as e:
            logger.error(f"Volatility mean reversion detection failed: {str(e)}")
            return {'signals': [], 'signal_count': 0, 'primary_signal': None}
    
    def analyze_volatility_term_structure(self, iv_data_by_expiry: Dict[str, float]) -> Dict[str, Any]:
        """Analyze implied volatility term structure."""
        try:
            if not iv_data_by_expiry or len(iv_data_by_expiry) < 2:
                return {'term_structure': 'insufficient_data'}
            
            # Sort by expiry (assuming expiry format allows sorting)
            sorted_expiries = sorted(iv_data_by_expiry.items())
            
            if len(sorted_expiries) < 2:
                return {'term_structure': 'insufficient_data'}
            
            # Analyze term structure shape
            iv_values = [iv for _, iv in sorted_expiries]
            
            # Calculate slopes between consecutive expiries
            slopes = []
            for i in range(1, len(iv_values)):
                slope = iv_values[i] - iv_values[i-1]
                slopes.append(slope)
            
            avg_slope = np.mean(slopes) if slopes else 0
            
            # Classify term structure
            if avg_slope > 0.02:  # Steep upward slope
                structure_type = 'steep_contango'
                trading_signal = 'calendar_spread_opportunity'
            elif avg_slope < -0.02:  # Steep downward slope
                structure_type = 'steep_backwardation'
                trading_signal = 'reverse_calendar_opportunity'
            elif abs(avg_slope) < 0.005:  # Flat structure
                structure_type = 'flat'
                trading_signal = 'neutral'
            else:
                structure_type = 'normal_contango' if avg_slope > 0 else 'mild_backwardation'
                trading_signal = 'monitor'
            
            # Front-end vs back-end comparison
            front_iv = iv_values[0]
            back_iv = iv_values[-1]
            front_back_spread = back_iv - front_iv
            
            return {
                'term_structure': structure_type,
                'trading_signal': trading_signal,
                'average_slope': avg_slope,
                'front_back_spread': front_back_spread,
                'front_iv': front_iv,
                'back_iv': back_iv,
                'iv_curve': iv_values
            }
            
        except Exception as e:
            logger.error(f"Volatility term structure analysis failed: {str(e)}")
            return {'term_structure': 'error'}
    
    def generate_iv_rank_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on IV rank analysis."""
        try:
            current_iv = data.get('current_iv', 0)
            historical_iv = data.get('historical_iv', [])
            current_hv = data.get('current_hv', 0)
            historical_data = data.get('historical_data', [])
            iv_by_expiry = data.get('iv_by_expiry', {})
            
            if current_iv <= 0 or not historical_iv:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'no_data',
                    'reasoning': 'Insufficient IV data'
                }
            
            # Calculate IV rank
            iv_rank_data = self.calculate_iv_rank(current_iv, historical_iv)
            
            # Analyze HV-IV relationship
            hv_iv_analysis = self.analyze_hv_iv_relationship(current_iv, current_hv, historical_data)
            
            # Detect mean reversion signals
            mean_reversion = self.detect_volatility_mean_reversion_signals(iv_rank_data, historical_data)
            
            # Analyze term structure
            term_structure = self.analyze_volatility_term_structure(iv_by_expiry)
            
            # Generate primary signal
            action = "hold"
            confidence = 0.0
            signal_type = "neutral"
            reasoning = []
            
            # Primary signal from mean reversion analysis
            primary_signal = mean_reversion.get('primary_signal')
            if primary_signal:
                if primary_signal['direction'] == 'sell_volatility':
                    action = "sell"
                    signal_type = "sell_high_iv"
                elif primary_signal['direction'] == 'buy_volatility':
                    action = "buy"
                    signal_type = "buy_low_iv"
                
                confidence += primary_signal['confidence']
                reasoning.append(primary_signal['reasoning'])
            
            # HV-IV divergence signals
            if hv_iv_analysis.get('divergence_detected'):
                divergence_type = hv_iv_analysis.get('divergence_type', '')
                
                if divergence_type == 'iv_premium':
                    if action == "hold":
                        action = "sell"
                        signal_type = "iv_overpriced"
                    confidence += 0.3
                    reasoning.append("IV trading at premium to HV")
                    
                elif divergence_type == 'iv_discount':
                    if action == "hold":
                        action = "buy"
                        signal_type = "iv_underpriced"
                    confidence += 0.3
                    reasoning.append("IV trading at discount to HV")
            
            # Term structure signals
            if term_structure.get('trading_signal') == 'calendar_spread_opportunity':
                confidence += 0.2
                reasoning.append("Steep contango in vol term structure")
            elif term_structure.get('trading_signal') == 'reverse_calendar_opportunity':
                confidence += 0.2
                reasoning.append("Steep backwardation in vol term structure")
            
            # Multiple signals boost confidence
            signal_count = mean_reversion.get('signal_count', 0)
            if signal_count > 1:
                confidence *= 1.2
                reasoning.append(f"Multiple volatility signals ({signal_count})")
            
            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"IV rank signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process volatility data and generate IV rank signals."""
        try:
            if "current_iv" not in data or "historical_iv" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Current IV or historical IV data not provided"}
                }
            
            # Generate signals
            signal_data = self.generate_iv_rank_signals(data)
            
            # Get comprehensive analysis
            current_iv = data["current_iv"]
            historical_iv = data["historical_iv"]
            current_hv = data.get("current_hv", 0)
            historical_data = data.get("historical_data", [])
            iv_by_expiry = data.get("iv_by_expiry", {})
            
            iv_rank_data = self.calculate_iv_rank(current_iv, historical_iv)
            hv_iv_analysis = self.analyze_hv_iv_relationship(current_iv, current_hv, historical_data)
            mean_reversion = self.detect_volatility_mean_reversion_signals(iv_rank_data, historical_data)
            term_structure = self.analyze_volatility_term_structure(iv_by_expiry)
            
            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "iv_rank_data": iv_rank_data,
                    "hv_iv_analysis": hv_iv_analysis,
                    "mean_reversion_signals": mean_reversion,
                    "term_structure": term_structure
                }
            }
            
        except Exception as e:
            logger.error(f"IV rank signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 