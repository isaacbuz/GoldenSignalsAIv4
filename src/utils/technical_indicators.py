"""
Technical Indicators Module
Provides comprehensive technical analysis indicators
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Calculate various technical indicators for trading analysis"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: List of closing prices
            period: Period for RSI calculation (default: 14)
            
        Returns:
            RSI value between 0 and 100
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        # Convert to pandas Series for easier calculation
        price_series = pd.Series(prices)
        
        # Calculate price differences
        delta = price_series.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Return the most recent RSI value
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: List of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if len(prices) < slow + signal:
            return {
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0
            }
        
        price_series = pd.Series(prices)
        
        # Calculate EMAs
        ema_fast = price_series.ewm(span=fast, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: List of closing prices
            period: Period for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        if len(prices) < period:
            current_price = prices[-1] if prices else 100
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98
            }
        
        price_series = pd.Series(prices)
        
        # Calculate middle band (SMA)
        middle = price_series.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = price_series.rolling(window=period).std()
        
        # Calculate bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else prices[-1] * 1.02,
            'middle': float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else prices[-1],
            'lower': float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else prices[-1] * 0.98
        }
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            prices: List of closing prices
            period: Period for SMA calculation
            
        Returns:
            SMA value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        return float(np.mean(prices[-period:]))
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            prices: List of closing prices
            period: Period for EMA calculation
            
        Returns:
            EMA value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        price_series = pd.Series(prices)
        ema = price_series.ewm(span=period, adjust=False).mean()
        
        return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else prices[-1]
    
    @staticmethod
    def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of closing prices
            period: Period for ATR calculation (default: 14)
            
        Returns:
            ATR value
        """
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return 0.0
        
        # Convert to pandas Series
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        # Calculate True Range
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift())
        tr3 = abs(low_series - close_series.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_volume_indicators(volumes: List[float], prices: List[float]) -> Dict[str, float]:
        """
        Calculate volume-based indicators
        
        Args:
            volumes: List of volume data
            prices: List of closing prices
            
        Returns:
            Dictionary with volume indicators
        """
        if len(volumes) < 20 or len(prices) < 20:
            return {
                'volume_sma': volumes[-1] if volumes else 0,
                'volume_ratio': 1.0,
                'obv': 0.0
            }
        
        volume_series = pd.Series(volumes)
        price_series = pd.Series(prices)
        
        # Volume SMA
        volume_sma = volume_series.rolling(window=20).mean()
        
        # Volume ratio (current vs average)
        volume_ratio = volumes[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        
        # On-Balance Volume (OBV)
        price_diff = price_series.diff()
        obv_direction = price_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume_series * obv_direction).cumsum()
        
        return {
            'volume_sma': float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else volumes[-1],
            'volume_ratio': float(volume_ratio),
            'obv': float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0.0
        }
    
    @staticmethod
    def identify_support_resistance(prices: List[float], period: int = 20) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels
        
        Args:
            prices: List of closing prices
            period: Period for analysis (default: 20)
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(prices) < period:
            current_price = prices[-1] if prices else 100
            return {
                'support': [current_price * 0.98, current_price * 0.96, current_price * 0.94],
                'resistance': [current_price * 1.02, current_price * 1.04, current_price * 1.06]
            }
        
        price_series = pd.Series(prices)
        
        # Find local minima and maxima
        rolling_min = price_series.rolling(window=period, center=True).min()
        rolling_max = price_series.rolling(window=period, center=True).max()
        
        # Identify support levels (local minima)
        support_mask = price_series == rolling_min
        support_levels = price_series[support_mask].unique()
        support_levels = sorted(support_levels)[:3]  # Top 3 support levels
        
        # Identify resistance levels (local maxima)
        resistance_mask = price_series == rolling_max
        resistance_levels = price_series[resistance_mask].unique()
        resistance_levels = sorted(resistance_levels, reverse=True)[:3]  # Top 3 resistance levels
        
        # Ensure we have at least 3 levels
        current_price = prices[-1]
        while len(support_levels) < 3:
            support_levels.append(current_price * (0.98 - len(support_levels) * 0.02))
        while len(resistance_levels) < 3:
            resistance_levels.append(current_price * (1.02 + len(resistance_levels) * 0.02))
        
        return {
            'support': [float(x) for x in support_levels],
            'resistance': [float(x) for x in resistance_levels]
        }
    
    @staticmethod
    def calculate_all_indicators(data: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate all technical indicators from OHLCV data
        
        Args:
            data: List of dictionaries with 'open', 'high', 'low', 'close', 'volume' keys
            
        Returns:
            Dictionary with all calculated indicators
        """
        if not data or len(data) < 2:
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'bb_upper': 100.0,
                'bb_middle': 100.0,
                'bb_lower': 100.0,
                'sma_20': 100.0,
                'sma_50': 100.0,
                'ema_12': 100.0,
                'ema_26': 100.0,
                'atr': 2.0,
                'volume_sma': 1000000,
                'volume_ratio': 1.0,
                'obv': 0.0,
                'support_levels': [98.0, 96.0, 94.0],
                'resistance_levels': [102.0, 104.0, 106.0],
                'trend': 'neutral'
            }
        
        # Extract price and volume data
        closes = [d.get('close', 0) for d in data]
        highs = [d.get('high', 0) for d in data]
        lows = [d.get('low', 0) for d in data]
        volumes = [d.get('volume', 0) for d in data]
        
        # Calculate indicators
        rsi = TechnicalIndicators.calculate_rsi(closes)
        macd_data = TechnicalIndicators.calculate_macd(closes)
        bb_data = TechnicalIndicators.calculate_bollinger_bands(closes)
        volume_data = TechnicalIndicators.calculate_volume_indicators(volumes, closes)
        support_resistance = TechnicalIndicators.identify_support_resistance(closes)
        
        # Calculate moving averages
        sma_20 = TechnicalIndicators.calculate_sma(closes, 20)
        sma_50 = TechnicalIndicators.calculate_sma(closes, 50)
        ema_12 = TechnicalIndicators.calculate_ema(closes, 12)
        ema_26 = TechnicalIndicators.calculate_ema(closes, 26)
        
        # Calculate ATR
        atr = TechnicalIndicators.calculate_atr(highs, lows, closes)
        
        # Determine trend
        if len(closes) >= 20:
            recent_sma = TechnicalIndicators.calculate_sma(closes[-20:], 10)
            older_sma = TechnicalIndicators.calculate_sma(closes[-40:-20], 10) if len(closes) >= 40 else recent_sma
            
            if recent_sma > older_sma * 1.02:
                trend = 'bullish'
            elif recent_sma < older_sma * 0.98:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
        
        return {
            'rsi': rsi,
            'macd': macd_data['macd'],
            'macd_signal': macd_data['signal'],
            'macd_histogram': macd_data['histogram'],
            'bb_upper': bb_data['upper'],
            'bb_middle': bb_data['middle'],
            'bb_lower': bb_data['lower'],
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'atr': atr,
            'volume_sma': volume_data['volume_sma'],
            'volume_ratio': volume_data['volume_ratio'],
            'obv': volume_data['obv'],
            'support_levels': support_resistance['support'],
            'resistance_levels': support_resistance['resistance'],
            'trend': trend
        } 