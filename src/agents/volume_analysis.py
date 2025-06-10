"""
Volume Analysis Agent - GoldenSignalsAI V3

Analyzes volume patterns, volume-price relationships, and volume-based
indicators to generate trading signals.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseAgent
from ..models.signals import Signal, SignalType, SignalStrength


class VolumeAnalysisAgent(BaseAgent):
    """
    Agent specializing in volume-based trading strategies
    """
    
    def __init__(self, name: str, db_manager, redis_manager):
        super().__init__(name, db_manager, redis_manager)
        self.volume_threshold = 1.5  # 50% above average volume
    
    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Signal]:
        """
        Perform volume analysis on the given symbol
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            historical_data: Historical price and volume data
            
        Returns:
            Volume-based signal or None
        """
        try:
            if not historical_data or len(historical_data) < 50:
                logger.warning(f"Insufficient historical data for {symbol} volume analysis")
                return None
            
            # Convert to DataFrame
            df = self._prepare_dataframe(historical_data)
            
            # Calculate volume indicators
            volume_data = await self._calculate_volume_indicators(df)
            
            # Generate signal
            signal = await self._generate_volume_signal(symbol, market_data, volume_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Volume analysis failed for {symbol}: {str(e)}")
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
                    'low': float(point.get('low', point.get('price', 0))),
                    'open': float(point.get('open', point.get('price', 0)))
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare DataFrame: {str(e)}")
            raise
    
    async def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            indicators = {}
            
            # Basic volume metrics
            for period in [5, 10, 20, 50]:
                avg_volume = df['volume'].rolling(window=period).mean()
                indicators[f'avg_volume_{period}d'] = avg_volume
                indicators[f'volume_ratio_{period}d'] = df['volume'] / avg_volume
            
            # Volume Moving Averages
            indicators['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            indicators['volume_ema_10'] = df['volume'].ewm(span=10).mean()
            
            # Volume Rate of Change
            for period in [1, 5, 10]:
                volume_roc = ((df['volume'] - df['volume'].shift(period)) / df['volume'].shift(period)) * 100
                indicators[f'volume_roc_{period}d'] = volume_roc
            
            # On-Balance Volume (OBV)
            obv = self._calculate_obv(df)
            indicators['obv'] = obv
            indicators['obv_sma_10'] = obv.rolling(window=10).mean()
            indicators['obv_sma_20'] = obv.rolling(window=20).mean()
            
            # Volume Weighted Average Price (VWAP)
            vwap = self._calculate_vwap(df)
            indicators['vwap'] = vwap
            indicators['price_vs_vwap'] = (df['close'] / vwap - 1) * 100
            
            # Accumulation/Distribution Line
            ad_line = self._calculate_ad_line(df)
            indicators['ad_line'] = ad_line
            indicators['ad_line_sma_10'] = ad_line.rolling(window=10).mean()
            
            # Chaikin Money Flow
            cmf = self._calculate_cmf(df, 20)
            indicators['cmf'] = cmf
            
            # Volume Oscillator
            vol_osc = self._calculate_volume_oscillator(df, 5, 10)
            indicators['volume_oscillator'] = vol_osc
            
            # Price Volume Trend (PVT)
            pvt = self._calculate_pvt(df)
            indicators['pvt'] = pvt
            indicators['pvt_sma_10'] = pvt.rolling(window=10).mean()
            
            # Klinger Oscillator
            klinger = self._calculate_klinger_oscillator(df)
            indicators['klinger'] = klinger
            
            # Volume-Price Correlation
            for period in [10, 20]:
                price_change = df['close'].pct_change()
                volume_change = df['volume'].pct_change()
                correlation = price_change.rolling(window=period).corr(volume_change)
                indicators[f'volume_price_corr_{period}d'] = correlation
            
            # Volume Breakouts
            for period in [10, 20]:
                volume_breakout = df['volume'] > (df['volume'].rolling(window=period).mean() * 2)
                indicators[f'volume_breakout_{period}d'] = volume_breakout.astype(int)
            
            # Ease of Movement
            eom = self._calculate_ease_of_movement(df, 14)
            indicators['ease_of_movement'] = eom
            
            # Volume Disparity Index
            vdi = self._calculate_volume_disparity_index(df, 14)
            indicators['volume_disparity_index'] = vdi
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate volume indicators: {str(e)}")
            raise
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_volume = df['volume'].cumsum()
        cumulative_vp = (typical_price * df['volume']).cumsum()
        
        return cumulative_vp / cumulative_volume
    
    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)  # Handle division by zero
        mfv = mfm * df['volume']
        return mfv.cumsum()
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)  # Handle division by zero
        mfv = mfm * df['volume']
        
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf.fillna(0)
    
    def _calculate_volume_oscillator(self, df: pd.DataFrame, short_period: int, long_period: int) -> pd.Series:
        """Calculate Volume Oscillator"""
        short_ma = df['volume'].rolling(window=short_period).mean()
        long_ma = df['volume'].rolling(window=long_period).mean()
        return ((short_ma - long_ma) / long_ma) * 100
    
    def _calculate_pvt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Price Volume Trend"""
        price_change_ratio = df['close'].pct_change()
        pvt = (price_change_ratio * df['volume']).cumsum()
        return pvt.fillna(0)
    
    def _calculate_klinger_oscillator(self, df: pd.DataFrame, fast_period: int = 34, slow_period: int = 55) -> pd.Series:
        """Calculate Klinger Oscillator (simplified version)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume_force = df['volume'] * typical_price.diff().apply(lambda x: 1 if x > 0 else -1)
        
        fast_ema = volume_force.ewm(span=fast_period).mean()
        slow_ema = volume_force.ewm(span=slow_period).mean()
        
        return fast_ema - slow_ema
    
    def _calculate_ease_of_movement(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Ease of Movement"""
        distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
        box_height = df['high'] - df['low']
        scale_factor = df['volume'] / (box_height * 1000000)  # Scale to millions
        
        eom = distance_moved / scale_factor.replace(0, np.nan)
        return eom.rolling(window=period).mean().fillna(0)
    
    def _calculate_volume_disparity_index(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Disparity Index"""
        volume_ma = df['volume'].rolling(window=period).mean()
        return ((df['volume'] - volume_ma) / volume_ma) * 100
    
    async def _generate_volume_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        volume_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate trading signal based on volume analysis"""
        try:
            current_price = market_data.get('price', 0)
            current_volume = market_data.get('volume', 0)
            
            if current_price <= 0:
                return None
            
            volume_scores = {}
            
            # Volume Ratio Analysis
            volume_ratio_score = 0
            for period in [10, 20]:
                key = f'volume_ratio_{period}d'
                if key in volume_data and not volume_data[key].empty:
                    volume_ratio = volume_data[key].iloc[-1]
                    weight = 0.3 if period == 10 else 0.2
                    
                    if volume_ratio > 2.0:  # Very high volume
                        volume_ratio_score += weight * 0.8
                    elif volume_ratio > 1.5:  # High volume
                        volume_ratio_score += weight * 0.6
                    elif volume_ratio < 0.5:  # Low volume
                        volume_ratio_score -= weight * 0.3
            
            volume_scores['volume_ratio'] = volume_ratio_score
            
            # OBV Analysis
            obv_score = 0
            if 'obv' in volume_data and 'obv_sma_20' in volume_data:
                if not volume_data['obv'].empty and not volume_data['obv_sma_20'].empty:
                    current_obv = volume_data['obv'].iloc[-1]
                    obv_sma = volume_data['obv_sma_20'].iloc[-1]
                    
                    if current_obv > obv_sma:
                        obv_score = 0.25  # Bullish OBV
                    else:
                        obv_score = -0.15  # Bearish OBV
            
            volume_scores['obv'] = obv_score
            
            # VWAP Analysis
            vwap_score = 0
            if 'price_vs_vwap' in volume_data and not volume_data['price_vs_vwap'].empty:
                price_vs_vwap = volume_data['price_vs_vwap'].iloc[-1]
                
                if price_vs_vwap > 2:  # Price well above VWAP
                    vwap_score = -0.2  # Potentially overvalued
                elif price_vs_vwap > 0:  # Price above VWAP
                    vwap_score = 0.15  # Bullish
                elif price_vs_vwap < -2:  # Price well below VWAP
                    vwap_score = 0.2  # Potentially undervalued
                else:  # Price below VWAP
                    vwap_score = -0.1  # Bearish
            
            volume_scores['vwap'] = vwap_score
            
            # Chaikin Money Flow Analysis
            cmf_score = 0
            if 'cmf' in volume_data and not volume_data['cmf'].empty:
                cmf = volume_data['cmf'].iloc[-1]
                
                if cmf > 0.2:  # Strong buying pressure
                    cmf_score = 0.3
                elif cmf > 0:  # Buying pressure
                    cmf_score = 0.15
                elif cmf < -0.2:  # Strong selling pressure
                    cmf_score = -0.3
                else:  # Selling pressure
                    cmf_score = -0.15
            
            volume_scores['cmf'] = cmf_score
            
            # Volume Oscillator Analysis
            vol_osc_score = 0
            if 'volume_oscillator' in volume_data and not volume_data['volume_oscillator'].empty:
                vol_osc = volume_data['volume_oscillator'].iloc[-1]
                
                if vol_osc > 20:  # High positive oscillator
                    vol_osc_score = 0.2
                elif vol_osc > 0:  # Positive oscillator
                    vol_osc_score = 0.1
                elif vol_osc < -20:  # High negative oscillator
                    vol_osc_score = -0.2
                else:  # Negative oscillator
                    vol_osc_score = -0.1
            
            volume_scores['volume_oscillator'] = vol_osc_score
            
            # A/D Line Analysis
            ad_score = 0
            if 'ad_line' in volume_data and 'ad_line_sma_10' in volume_data:
                if not volume_data['ad_line'].empty and not volume_data['ad_line_sma_10'].empty:
                    current_ad = volume_data['ad_line'].iloc[-1]
                    ad_sma = volume_data['ad_line_sma_10'].iloc[-1]
                    
                    if current_ad > ad_sma:
                        ad_score = 0.2  # Accumulation
                    else:
                        ad_score = -0.2  # Distribution
            
            volume_scores['ad_line'] = ad_score
            
            # Volume Breakout Analysis
            breakout_score = 0
            for period in [10, 20]:
                key = f'volume_breakout_{period}d'
                if key in volume_data and not volume_data[key].empty:
                    breakout = volume_data[key].iloc[-1]
                    weight = 0.15 if period == 10 else 0.1
                    
                    if breakout == 1:  # Volume breakout detected
                        breakout_score += weight
            
            volume_scores['breakout'] = breakout_score
            
            # Volume-Price Correlation Analysis
            correlation_score = 0
            for period in [10, 20]:
                key = f'volume_price_corr_{period}d'
                if key in volume_data and not volume_data[key].empty:
                    correlation = volume_data[key].iloc[-1]
                    weight = 0.1
                    
                    if correlation > 0.3:  # Strong positive correlation
                        correlation_score += weight
                    elif correlation < -0.3:  # Strong negative correlation
                        correlation_score -= weight * 0.5
            
            volume_scores['correlation'] = correlation_score
            
            # Calculate total volume score
            total_volume_score = sum(volume_scores.values())
            
            # Determine signal type and confidence
            if total_volume_score > 0.4:
                signal_type = SignalType.BUY
                confidence = min(total_volume_score, 0.9)
            elif total_volume_score < -0.4:
                signal_type = SignalType.SELL
                confidence = min(abs(total_volume_score), 0.9)
            else:
                signal_type = SignalType.HOLD
                confidence = 0.3
            
            # Boost confidence if volume is significantly high
            volume_ratio_20d = volume_data.get('volume_ratio_20d', pd.Series())
            if not volume_ratio_20d.empty:
                current_vol_ratio = volume_ratio_20d.iloc[-1]
                if current_vol_ratio > 2.0:  # Very high volume
                    confidence *= 1.2
                elif current_vol_ratio > 1.5:  # High volume
                    confidence *= 1.1
                elif current_vol_ratio < 0.7:  # Low volume
                    confidence *= 0.8
            
            confidence = max(0.1, min(0.9, confidence))
            
            # Determine signal strength
            if confidence >= 0.7:
                strength = SignalStrength.STRONG
            elif confidence >= 0.5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Generate reasoning
            reasoning = self._generate_volume_reasoning(signal_type, volume_scores, total_score)
            
            # Create signal
            signal = await self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    "total_volume_score": total_volume_score,
                    "volume_ratio": volume_scores.get('volume_ratio', 0),
                    "obv_signal": volume_scores.get('obv', 0),
                    "vwap_signal": volume_scores.get('vwap', 0),
                    "cmf_signal": volume_scores.get('cmf', 0),
                    "volume_oscillator": volume_scores.get('volume_oscillator', 0),
                    "ad_line_signal": volume_scores.get('ad_line', 0),
                    "breakout_signal": volume_scores.get('breakout', 0),
                    "correlation_signal": volume_scores.get('correlation', 0),
                    "current_volume_ratio": volume_data.get('volume_ratio_20d', pd.Series()).iloc[-1] if 'volume_ratio_20d' in volume_data else 0,
                    "current_cmf": volume_data.get('cmf', pd.Series()).iloc[-1] if 'cmf' in volume_data else 0
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate volume signal for {symbol}: {str(e)}")
            return None
    
    def _generate_volume_reasoning(
        self,
        signal_type: SignalType,
        volume_scores: Dict[str, float],
        total_score: float
    ) -> str:
        """Generate human-readable reasoning for volume signal"""
        try:
            reasons = []
            
            # Volume ratio reasoning
            volume_ratio = volume_scores.get('volume_ratio', 0)
            if volume_ratio > 0.3:
                reasons.append("High volume supporting price movement")
            elif volume_ratio < -0.1:
                reasons.append("Low volume weakening price movement")
            
            # OBV reasoning
            obv_signal = volume_scores.get('obv', 0)
            if obv_signal > 0.1:
                reasons.append("On-Balance Volume trending upward")
            elif obv_signal < -0.1:
                reasons.append("On-Balance Volume trending downward")
            
            # VWAP reasoning
            vwap_signal = volume_scores.get('vwap', 0)
            if vwap_signal > 0.1:
                reasons.append("Price above Volume Weighted Average Price")
            elif vwap_signal < -0.1:
                reasons.append("Price below Volume Weighted Average Price")
            
            # CMF reasoning
            cmf_signal = volume_scores.get('cmf', 0)
            if cmf_signal > 0.2:
                reasons.append("Strong buying pressure detected")
            elif cmf_signal < -0.2:
                reasons.append("Strong selling pressure detected")
            
            # Breakout reasoning
            breakout_signal = volume_scores.get('breakout', 0)
            if breakout_signal > 0.1:
                reasons.append("Volume breakout detected")
            
            if not reasons:
                reasons.append("Mixed volume signals")
            
            return f"Volume analysis {signal_type.value} (score: {total_score:.2f}): " + "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Failed to generate volume reasoning: {str(e)}")
            return f"Volume analysis {signal_type.value} signal" 