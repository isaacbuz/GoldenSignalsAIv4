"""
ADX (Average Directional Index) Agent
Measures trend strength regardless of direction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ADXAgent:
    """
    ADX trading agent for trend strength measurement
    
    Signals:
    - Strong/weak trend identification
    - Trend changes via +DI/-DI crossovers
    - Range-bound market detection
    """
    
    def __init__(self):
        self.name = "ADXAgent"
        self.adx_period = 14  # Standard ADX period
        self.strong_trend = 25  # ADX > 25 indicates strong trend
        self.weak_trend = 20   # ADX < 20 indicates weak/no trend
        
    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr
    
    def calculate_directional_movement(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate +DM and -DM"""
        up_move = df['High'] - df['High'].shift(1)
        down_move = df['Low'].shift(1) - df['Low']
        
        # +DM occurs when up_move > down_move and up_move > 0
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), 
                           index=df.index)
        
        # -DM occurs when down_move > up_move and down_move > 0
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), 
                            index=df.index)
        
        return plus_dm, minus_dm
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX, +DI, and -DI"""
        # Calculate True Range
        df['TR'] = self.calculate_true_range(df)
        
        # Calculate Directional Movements
        df['+DM'], df['-DM'] = self.calculate_directional_movement(df)
        
        # Smooth using Wilder's method (EMA with alpha = 1/period)
        alpha = 1 / self.adx_period
        
        # ATR (Average True Range)
        df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
        
        # Smoothed +DM and -DM
        df['+DM_smooth'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
        df['-DM_smooth'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate Directional Indicators
        df['+DI'] = 100 * df['+DM_smooth'] / df['ATR']
        df['-DI'] = 100 * df['-DM_smooth'] / df['ATR']
        
        # Calculate DX
        df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        
        # Calculate ADX (smoothed DX)
        df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
        
        return df
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate ADX trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty or len(df) < self.adx_period * 3:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")
            
            # Calculate ADX
            df = self.calculate_adx(df)
            
            # Get current values
            current_adx = df['ADX'].iloc[-1]
            current_plus_di = df['+DI'].iloc[-1]
            current_minus_di = df['-DI'].iloc[-1]
            prev_plus_di = df['+DI'].iloc[-2]
            prev_minus_di = df['-DI'].iloc[-2]
            
            # ADX trend (rising/falling)
            adx_change = current_adx - df['ADX'].iloc[-5]
            
            signals = []
            strength = 0
            
            # 1. Trend Strength Analysis
            if current_adx > self.strong_trend:
                if current_adx > 40:
                    signals.append("Very strong trend")
                    trend_multiplier = 1.3
                else:
                    signals.append("Strong trend")
                    trend_multiplier = 1.2
            elif current_adx < self.weak_trend:
                signals.append("Weak/No trend")
                trend_multiplier = 0.5
            else:
                signals.append("Moderate trend")
                trend_multiplier = 1.0
            
            # 2. Directional Analysis
            if current_plus_di > current_minus_di:
                signals.append("+DI > -DI (Bullish)")
                base_strength = 0.3
                
                # Check for crossover
                if prev_plus_di <= prev_minus_di:
                    signals.append("Bullish DI crossover")
                    base_strength = 0.5
            else:
                signals.append("-DI > +DI (Bearish)")
                base_strength = -0.3
                
                # Check for crossover
                if prev_plus_di >= prev_minus_di:
                    signals.append("Bearish DI crossover")
                    base_strength = -0.5
            
            # Apply trend strength multiplier
            strength = base_strength * trend_multiplier
            
            # 3. ADX Momentum
            if adx_change > 2:
                signals.append("ADX rising (trend strengthening)")
                strength *= 1.1
            elif adx_change < -2:
                signals.append("ADX falling (trend weakening)")
                strength *= 0.9
            
            # 4. Special Conditions
            if current_adx < 20 and abs(current_plus_di - current_minus_di) < 5:
                signals.append("Range-bound market")
                strength = 0  # Neutral in ranging markets
            
            # Price confirmation
            current_price = df['Close'].iloc[-1]
            sma_20 = df['Close'].tail(20).mean()
            
            if strength > 0 and current_price > sma_20:
                signals.append("Price confirms uptrend")
                strength += 0.1
            elif strength < 0 and current_price < sma_20:
                signals.append("Price confirms downtrend")
                strength -= 0.1
            
            # Determine action
            if strength >= 0.3:
                action = "BUY"
            elif strength <= -0.3:
                action = "SELL"
            else:
                action = "NEUTRAL"
            
            confidence = min(abs(strength), 1.0)
            
            return self._create_signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=f"ADX: {', '.join(signals)}",
                data={
                    "adx": float(current_adx),
                    "+di": float(current_plus_di),
                    "-di": float(current_minus_di),
                    "trend_strength": "Very Strong" if current_adx > 40 else 
                                     "Strong" if current_adx > 25 else 
                                     "Moderate" if current_adx > 20 else "Weak",
                    "adx_change": float(adx_change),
                    "signals": signals
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating ADX signal for {symbol}: {str(e)}")
            return self._create_signal(symbol, "ERROR", 0, str(e))
    
    def _create_signal(self, symbol: str, action: str, confidence: float, 
                      reason: str, data: Dict = None) -> Dict[str, Any]:
        """Create standardized signal output"""
        return {
            "agent": self.name,
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        } 