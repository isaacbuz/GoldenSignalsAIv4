"""
Mean Reversion Agent - GoldenSignalsAI V3

Identifies mean reversion trading opportunities by detecting when prices
deviate significantly from their historical means and are likely to revert.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseAgent
from ..models.signals import Signal, SignalType, SignalStrength


class MeanReversionAgent(BaseAgent):
    """
    Agent specializing in mean reversion trading strategies
    """
    
    def __init__(self, name: str, db_manager, redis_manager):
        super().__init__(name, db_manager, redis_manager)
        self.lookback_periods = [10, 20, 50, 100]
        self.reversion_threshold = 2.0  # Standard deviations
    
    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Signal]:
        """
        Perform mean reversion analysis on the given symbol
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            historical_data: Historical price data
            
        Returns:
            Mean reversion signal or None
        """
        try:
            if not historical_data or len(historical_data) < 100:
                logger.warning(f"Insufficient historical data for {symbol} mean reversion analysis")
                return None
            
            # Convert to DataFrame
            df = self._prepare_dataframe(historical_data)
            
            # Calculate mean reversion indicators
            reversion_data = await self._calculate_reversion_indicators(df)
            
            # Generate signal
            signal = await self._generate_reversion_signal(symbol, market_data, reversion_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Mean reversion analysis failed for {symbol}: {str(e)}")
            return None
    
    def _prepare_dataframe(self, historical_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        try:
            data = []
            for point in historical_data:
                data.append({
                    'timestamp': point.get('timestamp', datetime.utcnow()),
                    'close': float(point.get('close', point.get('price', 0))),
                    'volume': int(point.get('volume', 0)),
                    'high': float(point.get('high', point.get('price', 0))),
                    'low': float(point.get('low', point.get('price', 0)))
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare DataFrame: {str(e)}")
            raise
    
    async def _calculate_reversion_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mean reversion indicators"""
        try:
            indicators = {}
            
            # Z-scores for different periods
            for period in self.lookback_periods:
                mean = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                z_score = (df['close'] - mean) / std
                indicators[f'z_score_{period}d'] = z_score
                indicators[f'mean_{period}d'] = mean
                indicators[f'std_{period}d'] = std
            
            # Percent distance from moving averages
            for period in [10, 20, 50]:
                sma = df['close'].rolling(window=period).mean()
                distance_pct = ((df['close'] - sma) / sma) * 100
                indicators[f'distance_sma_{period}d'] = distance_pct
            
            # Bollinger Bands for mean reversion
            for period in [20, 50]:
                bb_mean = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                upper_band = bb_mean + (2 * bb_std)
                lower_band = bb_mean - (2 * bb_std)
                
                # Percentage of Bollinger Band position
                bb_position = (df['close'] - lower_band) / (upper_band - lower_band)
                indicators[f'bb_position_{period}d'] = bb_position
                indicators[f'bb_upper_{period}d'] = upper_band
                indicators[f'bb_lower_{period}d'] = lower_band
                indicators[f'bb_mean_{period}d'] = bb_mean
                
                # Bollinger Band squeeze (low volatility)
                bb_width = (upper_band - lower_band) / bb_mean
                indicators[f'bb_width_{period}d'] = bb_width
            
            # RSI for overbought/oversold conditions
            rsi = self._calculate_rsi(df['close'], 14)
            indicators['rsi'] = rsi
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # Williams %R
            williams_r = self._calculate_williams_r(df, 14)
            indicators['williams_r'] = williams_r
            
            # Price channel analysis
            for period in [10, 20]:
                highest_high = df['high'].rolling(window=period).max()
                lowest_low = df['low'].rolling(window=period).min()
                channel_position = (df['close'] - lowest_low) / (highest_high - lowest_low)
                indicators[f'channel_position_{period}d'] = channel_position
            
            # Volatility analysis (for mean reversion timing)
            for period in [10, 20]:
                volatility = df['close'].pct_change().rolling(window=period).std() * np.sqrt(252)
                indicators[f'volatility_{period}d'] = volatility
            
            # Reversion strength (how often price reverts)
            for period in [5, 10]:
                reversion_count = 0
                for i in range(period, len(df)):
                    if i < period:
                        continue
                    price_series = df['close'].iloc[i-period:i]
                    if len(price_series) >= 3:
                        # Check if price reverted to mean in the period
                        mean_price = price_series.mean()
                        first_half_avg = price_series.iloc[:period//2].mean()
                        second_half_avg = price_series.iloc[period//2:].mean()
                        
                        # If price moved away from mean then back
                        if abs(first_half_avg - mean_price) > abs(second_half_avg - mean_price):
                            reversion_count += 1
                
                reversion_strength = reversion_count / max(1, len(df) - period)
                indicators[f'reversion_strength_{period}d'] = pd.Series([reversion_strength] * len(df), index=df.index)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate reversion indicators: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int) -> tuple:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        k_percent = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        return ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
    
    async def _generate_reversion_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        reversion_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate trading signal based on mean reversion analysis"""
        try:
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None
            
            reversion_scores = {}
            
            # Z-score analysis
            z_score_signal = 0
            for period in self.lookback_periods:
                key = f'z_score_{period}d'
                if key in reversion_data and not reversion_data[key].empty:
                    z_score = reversion_data[key].iloc[-1]
                    weight = 0.3 if period <= 20 else 0.2
                    
                    if z_score > self.reversion_threshold:  # Overbought, expect reversion down
                        z_score_signal -= weight
                    elif z_score < -self.reversion_threshold:  # Oversold, expect reversion up
                        z_score_signal += weight
            
            reversion_scores['z_score'] = z_score_signal
            
            # Distance from moving averages
            distance_signal = 0
            for period in [10, 20, 50]:
                key = f'distance_sma_{period}d'
                if key in reversion_data and not reversion_data[key].empty:
                    distance = reversion_data[key].iloc[-1]
                    weight = 0.25 if period <= 20 else 0.15
                    
                    if distance > 5:  # 5% above MA, expect reversion down
                        distance_signal -= weight
                    elif distance < -5:  # 5% below MA, expect reversion up
                        distance_signal += weight
            
            reversion_scores['distance'] = distance_signal
            
            # Bollinger Bands analysis
            bb_signal = 0
            for period in [20, 50]:
                position_key = f'bb_position_{period}d'
                width_key = f'bb_width_{period}d'
                
                if position_key in reversion_data and not reversion_data[position_key].empty:
                    bb_position = reversion_data[position_key].iloc[-1]
                    weight = 0.3 if period == 20 else 0.2
                    
                    if bb_position > 0.9:  # Near upper band, expect reversion down
                        bb_signal -= weight
                    elif bb_position < 0.1:  # Near lower band, expect reversion up
                        bb_signal += weight
                
                # Low volatility (squeeze) enhances mean reversion signals
                if width_key in reversion_data and not reversion_data[width_key].empty:
                    bb_width = reversion_data[width_key].iloc[-1]
                    avg_width = reversion_data[width_key].rolling(window=50).mean().iloc[-1]
                    
                    if bb_width < avg_width * 0.8:  # Low volatility
                        bb_signal *= 1.2  # Enhance signal strength
            
            reversion_scores['bollinger_bands'] = bb_signal
            
            # RSI overbought/oversold
            rsi_signal = 0
            if 'rsi' in reversion_data and not reversion_data['rsi'].empty:
                rsi = reversion_data['rsi'].iloc[-1]
                
                if rsi > 80:  # Extreme overbought
                    rsi_signal = -0.4
                elif rsi > 70:  # Overbought
                    rsi_signal = -0.25
                elif rsi < 20:  # Extreme oversold
                    rsi_signal = 0.4
                elif rsi < 30:  # Oversold
                    rsi_signal = 0.25
            
            reversion_scores['rsi'] = rsi_signal
            
            # Stochastic Oscillator
            stoch_signal = 0
            if 'stoch_k' in reversion_data and 'stoch_d' in reversion_data:
                if not reversion_data['stoch_k'].empty and not reversion_data['stoch_d'].empty:
                    stoch_k = reversion_data['stoch_k'].iloc[-1]
                    stoch_d = reversion_data['stoch_d'].iloc[-1]
                    
                    if stoch_k > 80 and stoch_d > 80:  # Overbought
                        stoch_signal = -0.2
                    elif stoch_k < 20 and stoch_d < 20:  # Oversold
                        stoch_signal = 0.2
            
            reversion_scores['stochastic'] = stoch_signal
            
            # Williams %R
            williams_signal = 0
            if 'williams_r' in reversion_data and not reversion_data['williams_r'].empty:
                williams_r = reversion_data['williams_r'].iloc[-1]
                
                if williams_r > -20:  # Overbought
                    williams_signal = -0.15
                elif williams_r < -80:  # Oversold
                    williams_signal = 0.15
            
            reversion_scores['williams_r'] = williams_signal
            
            # Channel position
            channel_signal = 0
            for period in [10, 20]:
                key = f'channel_position_{period}d'
                if key in reversion_data and not reversion_data[key].empty:
                    channel_pos = reversion_data[key].iloc[-1]
                    weight = 0.15
                    
                    if channel_pos > 0.9:  # Near top of channel
                        channel_signal -= weight
                    elif channel_pos < 0.1:  # Near bottom of channel
                        channel_signal += weight
            
            reversion_scores['channel'] = channel_signal
            
            # Calculate total reversion score
            total_reversion_score = sum(reversion_scores.values())
            
            # Determine signal type and confidence
            if total_reversion_score > 0.4:
                signal_type = SignalType.BUY  # Expect upward reversion
                confidence = min(total_reversion_score, 0.9)
            elif total_reversion_score < -0.4:
                signal_type = SignalType.SELL  # Expect downward reversion
                confidence = min(abs(total_reversion_score), 0.9)
            else:
                signal_type = SignalType.HOLD
                confidence = 0.3
            
            # Adjust confidence based on volatility
            volatility_key = 'volatility_20d'
            if volatility_key in reversion_data and not reversion_data[volatility_key].empty:
                volatility = reversion_data[volatility_key].iloc[-1]
                avg_volatility = reversion_data[volatility_key].rolling(window=50).mean().iloc[-1]
                
                if volatility > avg_volatility * 1.5:  # High volatility
                    confidence *= 0.8  # Reduce confidence in high volatility
                elif volatility < avg_volatility * 0.7:  # Low volatility
                    confidence *= 1.1  # Increase confidence in low volatility
            
            confidence = max(0.1, min(0.9, confidence))
            
            # Determine signal strength
            if confidence >= 0.7:
                strength = SignalStrength.STRONG
            elif confidence >= 0.5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Generate reasoning
            reasoning = self._generate_reversion_reasoning(signal_type, reversion_scores, total_reversion_score)
            
            # Create signal
            signal = await self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    "total_reversion_score": total_reversion_score,
                    "z_score_signal": reversion_scores.get('z_score', 0),
                    "distance_signal": reversion_scores.get('distance', 0),
                    "bb_signal": reversion_scores.get('bollinger_bands', 0),
                    "rsi_signal": reversion_scores.get('rsi', 0),
                    "stoch_signal": reversion_scores.get('stochastic', 0),
                    "williams_signal": reversion_scores.get('williams_r', 0),
                    "channel_signal": reversion_scores.get('channel', 0),
                    "current_rsi": reversion_data.get('rsi', pd.Series()).iloc[-1] if 'rsi' in reversion_data else 0,
                    "z_score_20d": reversion_data.get('z_score_20d', pd.Series()).iloc[-1] if 'z_score_20d' in reversion_data else 0
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate reversion signal for {symbol}: {str(e)}")
            return None
    
    def _generate_reversion_reasoning(
        self,
        signal_type: SignalType,
        reversion_scores: Dict[str, float],
        total_score: float
    ) -> str:
        """Generate human-readable reasoning for mean reversion signal"""
        try:
            reasons = []
            
            # Z-score reasoning
            z_score = reversion_scores.get('z_score', 0)
            if z_score > 0.2:
                reasons.append("Price significantly below statistical mean")
            elif z_score < -0.2:
                reasons.append("Price significantly above statistical mean")
            
            # Distance reasoning
            distance = reversion_scores.get('distance', 0)
            if distance > 0.2:
                reasons.append("Price extended below moving averages")
            elif distance < -0.2:
                reasons.append("Price extended above moving averages")
            
            # Bollinger Bands reasoning
            bb_signal = reversion_scores.get('bollinger_bands', 0)
            if bb_signal > 0.2:
                reasons.append("Price near lower Bollinger Band")
            elif bb_signal < -0.2:
                reasons.append("Price near upper Bollinger Band")
            
            # RSI reasoning
            rsi_signal = reversion_scores.get('rsi', 0)
            if rsi_signal > 0.2:
                reasons.append("RSI indicates oversold conditions")
            elif rsi_signal < -0.2:
                reasons.append("RSI indicates overbought conditions")
            
            # Stochastic reasoning
            stoch_signal = reversion_scores.get('stochastic', 0)
            if stoch_signal > 0.1:
                reasons.append("Stochastic oscillator oversold")
            elif stoch_signal < -0.1:
                reasons.append("Stochastic oscillator overbought")
            
            if not reasons:
                reasons.append("Mixed mean reversion signals")
            
            return f"Mean reversion analysis {signal_type.value} (score: {total_score:.2f}): " + "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Failed to generate reversion reasoning: {str(e)}")
            return f"Mean reversion analysis {signal_type.value} signal" 