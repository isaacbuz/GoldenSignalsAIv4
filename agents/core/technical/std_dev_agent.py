"""
Standard Deviation Agent
Measures price volatility using statistical standard deviation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StandardDeviationAgent:
    """
    Standard Deviation trading agent for volatility analysis
    
    Signals:
    - High/low volatility periods
    - Volatility breakouts
    - Mean reversion opportunities
    """
    
    def __init__(self):
        self.name = "StandardDeviationAgent"
        self.lookback_period = 20  # Standard period for volatility
        self.volatility_window = 50  # For historical volatility comparison
        
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various volatility metrics"""
        # Standard deviation of returns
        df['Returns'] = df['Close'].pct_change()
        df['StdDev'] = df['Returns'].rolling(window=self.lookback_period).std()
        
        # Annualized volatility
        df['AnnualizedVol'] = df['StdDev'] * np.sqrt(252)
        
        # Price standard deviation
        df['PriceStdDev'] = df['Close'].rolling(window=self.lookback_period).std()
        
        # Z-score (distance from mean in standard deviations)
        rolling_mean = df['Close'].rolling(window=self.lookback_period).mean()
        df['ZScore'] = (df['Close'] - rolling_mean) / df['PriceStdDev']
        
        # Historical volatility percentile
        df['VolPercentile'] = df['StdDev'].rolling(window=self.volatility_window).rank(pct=True)
        
        return df
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Standard Deviation trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo", interval="1d")
            
            if df.empty or len(df) < self.volatility_window:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")
            
            # Calculate volatility metrics
            df = self.calculate_volatility_metrics(df)
            
            # Get current values
            current_price = df['Close'].iloc[-1]
            current_std = df['StdDev'].iloc[-1]
            current_price_std = df['PriceStdDev'].iloc[-1]
            current_zscore = df['ZScore'].iloc[-1]
            current_vol_percentile = df['VolPercentile'].iloc[-1]
            annualized_vol = df['AnnualizedVol'].iloc[-1]
            
            # Historical comparison
            avg_std = df['StdDev'].tail(self.volatility_window).mean()
            
            signals = []
            strength = 0
            
            # 1. Volatility Level Analysis
            if current_vol_percentile > 0.8:
                signals.append(f"High volatility (80th percentile)")
                volatility_state = "high"
            elif current_vol_percentile < 0.2:
                signals.append(f"Low volatility (20th percentile)")
                volatility_state = "low"
            else:
                signals.append(f"Normal volatility ({current_vol_percentile:.0%} percentile)")
                volatility_state = "normal"
            
            # 2. Z-Score Analysis (Mean Reversion)
            if pd.notna(current_zscore):
                if current_zscore > 2:
                    signals.append(f"Overbought (Z={current_zscore:.2f})")
                    strength -= 0.4  # Expect reversion
                elif current_zscore < -2:
                    signals.append(f"Oversold (Z={current_zscore:.2f})")
                    strength += 0.4  # Expect bounce
                elif abs(current_zscore) > 1.5:
                    signals.append(f"Extended from mean (Z={current_zscore:.2f})")
                    strength += -0.2 * np.sign(current_zscore)
                else:
                    signals.append(f"Near mean (Z={current_zscore:.2f})")
            
            # 3. Volatility Expansion/Contraction
            vol_change = (current_std - avg_std) / avg_std
            
            if vol_change > 0.5:
                signals.append("Volatility expanding rapidly")
                # High volatility often precedes reversals
                if current_zscore > 1:
                    strength -= 0.2
                elif current_zscore < -1:
                    strength += 0.2
            elif vol_change < -0.3:
                signals.append("Volatility contracting")
                # Low volatility often precedes breakouts
                strength += 0.1 * np.sign(current_zscore)
            
            # 4. Trend Analysis with Volatility
            sma_20 = df['Close'].tail(20).mean()
            price_position = (current_price - sma_20) / sma_20
            
            if volatility_state == "low":
                if price_position > 0.01:
                    signals.append("Uptrend in low volatility")
                    strength += 0.3
                elif price_position < -0.01:
                    signals.append("Downtrend in low volatility")
                    strength -= 0.3
            elif volatility_state == "high":
                # Be more cautious in high volatility
                strength *= 0.7
                signals.append("High volatility - reduced confidence")
            
            # 5. Volatility Breakout Detection
            recent_vol = df['StdDev'].tail(5).mean()
            if current_std > recent_vol * 1.5:
                signals.append("Volatility breakout")
                # Check direction
                recent_return = df['Returns'].iloc[-1]
                if recent_return > 0:
                    strength += 0.2
                else:
                    strength -= 0.2
            
            # 6. Risk-adjusted signal
            if annualized_vol > 0.4:  # 40% annualized volatility
                signals.append(f"Very high risk ({annualized_vol:.1%} annual vol)")
                strength *= 0.5
            elif annualized_vol < 0.1:  # 10% annualized volatility
                signals.append(f"Low risk ({annualized_vol:.1%} annual vol)")
                strength *= 1.2
            
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
                reason=f"StdDev: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "std_dev": float(current_std) if pd.notna(current_std) else None,
                    "price_std_dev": float(current_price_std) if pd.notna(current_price_std) else None,
                    "z_score": float(current_zscore) if pd.notna(current_zscore) else None,
                    "volatility_percentile": float(current_vol_percentile) if pd.notna(current_vol_percentile) else None,
                    "annualized_volatility": float(annualized_vol) if pd.notna(annualized_vol) else None,
                    "volatility_state": volatility_state,
                    "signals": signals
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating Standard Deviation signal for {symbol}: {str(e)}")
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