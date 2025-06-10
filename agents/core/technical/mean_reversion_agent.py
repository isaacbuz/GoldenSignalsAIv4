"""
Mean Reversion Agent - Identifies mean reversion opportunities using statistical measures.
Uses Z-score, Bollinger bands, RSI, and other indicators to detect oversold/overbought conditions.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging
from scipy import stats
from ..base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MeanReversionAgent(BaseAgent):
    """Agent that identifies mean reversion trading opportunities."""
    
    def __init__(
        self,
        name: str = "MeanReversion",
        lookback_period: int = 20,
        z_score_threshold: float = 2.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        min_reversion_threshold: float = 0.05
    ):
        """
        Initialize Mean Reversion agent.
        
        Args:
            name: Agent name
            lookback_period: Period for statistical calculations
            z_score_threshold: Z-score threshold for mean reversion
            bb_period: Bollinger bands period
            bb_std: Bollinger bands standard deviation
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            min_reversion_threshold: Minimum expected reversion (5%)
        """
        super().__init__(name=name, agent_type="technical")
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_reversion_threshold = min_reversion_threshold
        
    def calculate_zscore(self, prices: pd.Series) -> Optional[float]:
        """Calculate Z-score of current price relative to historical mean."""
        try:
            if len(prices) < self.lookback_period:
                return None
                
            recent_prices = prices.tail(self.lookback_period)
            mean_price = recent_prices.mean()
            std_price = recent_prices.std()
            
            if std_price == 0:
                return 0.0
                
            current_price = prices.iloc[-1]
            z_score = (current_price - mean_price) / std_price
            
            return float(z_score)
            
        except Exception as e:
            logger.error(f"Z-score calculation failed: {str(e)}")
            return None
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands and position within bands."""
        try:
            if len(prices) < self.bb_period:
                return None
                
            # Calculate moving average and standard deviation
            sma = prices.rolling(window=self.bb_period).mean()
            std = prices.rolling(window=self.bb_period).std()
            
            # Calculate bands
            upper_band = sma + (std * self.bb_std)
            lower_band = sma - (std * self.bb_std)
            
            current_price = prices.iloc[-1]
            current_sma = sma.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # Calculate position within bands (0 = lower band, 1 = upper band)
            if current_upper != current_lower:
                bb_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                bb_position = 0.5
                
            # Calculate bandwidth (volatility measure)
            bandwidth = (current_upper - current_lower) / current_sma
            
            return {
                'upper_band': current_upper,
                'lower_band': current_lower,
                'middle_band': current_sma,
                'bb_position': bb_position,
                'bandwidth': bandwidth,
                'squeeze': bandwidth < 0.10  # Low volatility squeeze
            }
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {str(e)}")
            return None
    
    def calculate_rsi(self, prices: pd.Series) -> Optional[float]:
        """Calculate RSI for momentum confirmation."""
        try:
            if len(prices) < self.rsi_period + 1:
                return None
                
            # Calculate price changes
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0.0)
            losses = -deltas.where(deltas < 0, 0.0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=self.rsi_period).mean()
            avg_loss = losses.rolling(window=self.rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {str(e)}")
            return None
    
    def calculate_price_channels(self, highs: pd.Series, lows: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate price channels for additional mean reversion signals."""
        try:
            if len(highs) < self.lookback_period or len(lows) < self.lookback_period:
                return None
                
            # Calculate channel boundaries
            channel_high = highs.rolling(window=self.lookback_period).max().iloc[-1]
            channel_low = lows.rolling(window=self.lookback_period).min().iloc[-1]
            channel_mid = (channel_high + channel_low) / 2
            
            current_price = (highs.iloc[-1] + lows.iloc[-1]) / 2
            
            # Calculate position within channel
            if channel_high != channel_low:
                channel_position = (current_price - channel_low) / (channel_high - channel_low)
            else:
                channel_position = 0.5
                
            return {
                'channel_high': channel_high,
                'channel_low': channel_low,
                'channel_mid': channel_mid,
                'channel_position': channel_position
            }
            
        except Exception as e:
            logger.error(f"Price channel calculation failed: {str(e)}")
            return None
    
    def detect_mean_reversion_setup(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect mean reversion setup using multiple indicators."""
        try:
            closes = pd.Series(data["close_prices"])
            highs = pd.Series(data.get("high_prices", closes))
            lows = pd.Series(data.get("low_prices", closes))
            
            # Calculate all indicators
            z_score = self.calculate_zscore(closes)
            bollinger_data = self.calculate_bollinger_bands(closes)
            rsi = self.calculate_rsi(closes)
            channel_data = self.calculate_price_channels(highs, lows)
            
            if z_score is None or bollinger_data is None:
                return None
            
            # Detect oversold conditions (buy signal)
            oversold_signals = []
            
            if z_score <= -self.z_score_threshold:
                oversold_signals.append(f"Z-score: {z_score:.2f}")
                
            if bollinger_data['bb_position'] <= 0.05:  # Near lower band
                oversold_signals.append(f"BB position: {bollinger_data['bb_position']:.2f}")
                
            if rsi is not None and rsi <= self.rsi_oversold:
                oversold_signals.append(f"RSI: {rsi:.2f}")
                
            if channel_data and channel_data['channel_position'] <= 0.1:
                oversold_signals.append(f"Channel position: {channel_data['channel_position']:.2f}")
            
            # Detect overbought conditions (sell signal)
            overbought_signals = []
            
            if z_score >= self.z_score_threshold:
                overbought_signals.append(f"Z-score: {z_score:.2f}")
                
            if bollinger_data['bb_position'] >= 0.95:  # Near upper band
                overbought_signals.append(f"BB position: {bollinger_data['bb_position']:.2f}")
                
            if rsi is not None and rsi >= self.rsi_overbought:
                overbought_signals.append(f"RSI: {rsi:.2f}")
                
            if channel_data and channel_data['channel_position'] >= 0.9:
                overbought_signals.append(f"Channel position: {channel_data['channel_position']:.2f}")
            
            # Determine signal
            signal_type = None
            confidence = 0.0
            signals = []
            
            if len(oversold_signals) >= 2:  # At least 2 confirming indicators
                signal_type = 'oversold'
                confidence = min(1.0, len(oversold_signals) / 4)  # Max when all 4 agree
                signals = oversold_signals
                
            elif len(overbought_signals) >= 2:  # At least 2 confirming indicators
                signal_type = 'overbought'
                confidence = min(1.0, len(overbought_signals) / 4)  # Max when all 4 agree
                signals = overbought_signals
            
            # Add volatility boost (higher volatility = better mean reversion opportunity)
            if bollinger_data['bandwidth'] > 0.15:  # High volatility
                confidence *= 1.2
            elif bollinger_data['squeeze']:  # Low volatility squeeze
                confidence *= 0.8
            
            if signal_type:
                return {
                    'signal_type': signal_type,
                    'confidence': min(1.0, confidence),
                    'confirming_signals': signals,
                    'z_score': z_score,
                    'bollinger_data': bollinger_data,
                    'rsi': rsi,
                    'channel_data': channel_data
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Mean reversion setup detection failed: {str(e)}")
            return None
    
    def estimate_reversion_target(self, data: Dict[str, Any], signal_type: str) -> Optional[float]:
        """Estimate price target for mean reversion."""
        try:
            closes = pd.Series(data["close_prices"])
            current_price = closes.iloc[-1]
            
            # Calculate multiple potential targets
            targets = []
            
            # Moving average target
            if len(closes) >= self.lookback_period:
                ma_target = closes.tail(self.lookback_period).mean()
                targets.append(ma_target)
            
            # Bollinger Bands middle target
            bollinger_data = self.calculate_bollinger_bands(closes)
            if bollinger_data:
                targets.append(bollinger_data['middle_band'])
            
            # Channel middle target
            highs = pd.Series(data.get("high_prices", closes))
            lows = pd.Series(data.get("low_prices", closes))
            channel_data = self.calculate_price_channels(highs, lows)
            if channel_data:
                targets.append(channel_data['channel_mid'])
            
            if not targets:
                return None
            
            # Use median target to avoid outliers
            target_price = np.median(targets)
            
            # Calculate expected reversion percentage
            reversion_pct = abs(target_price - current_price) / current_price
            
            # Only proceed if expected reversion meets minimum threshold
            if reversion_pct >= self.min_reversion_threshold:
                return target_price
            
            return None
            
        except Exception as e:
            logger.error(f"Reversion target estimation failed: {str(e)}")
            return None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate mean reversion signals."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")
            
            closes = pd.Series(data["close_prices"])
            
            if len(closes) < max(self.lookback_period, self.bb_period, self.rsi_period):
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data for mean reversion analysis"}
                }
            
            # Detect mean reversion setup
            setup = self.detect_mean_reversion_setup(data)
            
            if setup is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"no_reversion_setup": True}
                }
            
            # Generate action based on signal type
            if setup['signal_type'] == 'oversold':
                action = "buy"
            elif setup['signal_type'] == 'overbought':
                action = "sell"
            else:
                action = "hold"
            
            # Estimate reversion target
            target = self.estimate_reversion_target(data, setup['signal_type'])
            
            return {
                "action": action,
                "confidence": setup['confidence'],
                "metadata": {
                    "setup": setup,
                    "target_price": target,
                    "current_price": closes.iloc[-1],
                    "reversion_type": setup['signal_type']
                }
            }
            
        except Exception as e:
            logger.error(f"Mean reversion signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 