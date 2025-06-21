"""
Options Flow Agent
Analyzes options market dynamics using volatility and price action as proxies
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SimpleOptionsFlowAgent:
    """
    Options Flow trading agent (using proxies)
    
    Signals:
    - Implied volatility proxies
    - Put/Call ratio estimates
    - Gamma exposure patterns
    - Options expiration effects
    """
    
    def __init__(self):
        self.name = "SimpleOptionsFlowAgent"
        self.iv_lookback = 30  # Days for IV calculation
        
    def estimate_implied_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimate IV using historical volatility patterns"""
        returns = df['Close'].pct_change()
        
        # Different period volatilities
        hv_10 = returns.tail(10).std() * np.sqrt(252)
        hv_20 = returns.tail(20).std() * np.sqrt(252)
        hv_30 = returns.tail(30).std() * np.sqrt(252)
        
        # Estimate current IV (typically higher than HV)
        estimated_iv = hv_10 * 1.2  # Simple multiplier
        
        # IV rank (where is current vol vs history)
        vol_history = []
        for i in range(30, len(df)):
            period_vol = returns.iloc[i-30:i].std() * np.sqrt(252)
            vol_history.append(period_vol)
        
        if vol_history:
            iv_rank = (estimated_iv - min(vol_history)) / (max(vol_history) - min(vol_history)) * 100
        else:
            iv_rank = 50
        
        return {
            'estimated_iv': estimated_iv,
            'hv_10': hv_10,
            'hv_20': hv_20,
            'hv_30': hv_30,
            'iv_rank': iv_rank
        }
    
    def estimate_put_call_ratio(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimate put/call ratio using price action and volume"""
        recent = df.tail(20)
        
        # Down days with high volume = put buying
        # Up days with high volume = call buying
        put_volume = 0
        call_volume = 0
        
        avg_volume = recent['Volume'].mean()
        
        for _, row in recent.iterrows():
            if row['Volume'] > avg_volume * 0.8:  # Significant volume
                if row['Close'] < row['Open']:
                    # Down day - estimate put volume
                    put_volume += row['Volume'] * abs(row['Close'] - row['Open']) / row['Open']
                else:
                    # Up day - estimate call volume
                    call_volume += row['Volume'] * abs(row['Close'] - row['Open']) / row['Open']
        
        # Calculate ratio
        if call_volume > 0:
            pc_ratio = put_volume / call_volume
        else:
            pc_ratio = 1.0
        
        # Also calculate skew proxy
        # More downside moves = put demand
        down_moves = recent[recent['Close'] < recent['Open']]['Close'].pct_change().abs().mean()
        up_moves = recent[recent['Close'] > recent['Open']]['Close'].pct_change().abs().mean()
        
        if up_moves > 0:
            skew = down_moves / up_moves
        else:
            skew = 1.0
        
        return {
            'pc_ratio': pc_ratio,
            'skew': skew,
            'put_volume_est': put_volume,
            'call_volume_est': call_volume
        }
    
    def detect_gamma_exposure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential gamma exposure patterns"""
        # Gamma exposure shows up as:
        # 1. Price pinning near round numbers
        # 2. Increased volatility away from strikes
        # 3. Compression near expiration
        
        current_price = df['Close'].iloc[-1]
        
        # Find nearest round number (potential strike)
        round_strike = round(current_price / 5) * 5  # $5 strikes
        distance_to_strike = abs(current_price - round_strike) / current_price
        
        # Check for pinning behavior
        recent_5d = df.tail(5)
        price_range = (recent_5d['High'].max() - recent_5d['Low'].min()) / current_price
        
        # Detect if we're near options expiration (Friday)
        last_date = df.index[-1]
        days_to_friday = (4 - last_date.weekday()) % 7
        near_expiration = days_to_friday <= 1
        
        # Gamma exposure indicators
        pinning = distance_to_strike < 0.01 and price_range < 0.02
        
        return {
            'nearest_strike': round_strike,
            'distance_to_strike': distance_to_strike,
            'price_range_5d': price_range,
            'pinning_detected': pinning,
            'near_expiration': near_expiration,
            'days_to_friday': days_to_friday
        }
    
    def analyze_volatility_term_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility term structure patterns"""
        returns = df['Close'].pct_change()
        
        # Short vs long term volatility
        vol_5d = returns.tail(5).std() * np.sqrt(252)
        vol_20d = returns.tail(20).std() * np.sqrt(252)
        vol_60d = returns.tail(60).std() * np.sqrt(252) if len(df) > 60 else vol_20d
        
        # Term structure slope
        if vol_60d > 0:
            term_structure = vol_5d / vol_60d
        else:
            term_structure = 1.0
        
        # Volatility of volatility
        vol_changes = []
        for i in range(20, len(df)):
            period_vol = returns.iloc[i-10:i].std()
            vol_changes.append(period_vol)
        
        if len(vol_changes) > 1:
            vol_of_vol = np.std(vol_changes)
        else:
            vol_of_vol = 0
        
        return {
            'vol_5d': vol_5d,
            'vol_20d': vol_20d,
            'vol_60d': vol_60d,
            'term_structure': term_structure,
            'vol_of_vol': vol_of_vol
        }
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Options Flow trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty or len(df) < 60:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")
            
            # Calculate options flow proxies
            iv_data = self.estimate_implied_volatility(df)
            pc_data = self.estimate_put_call_ratio(df)
            gamma_data = self.detect_gamma_exposure(df)
            term_data = self.analyze_volatility_term_structure(df)
            
            current_price = df['Close'].iloc[-1]
            sma_20 = df['Close'].tail(20).mean()
            
            signals = []
            strength = 0
            
            # 1. Implied Volatility Analysis
            iv_rank = iv_data['iv_rank']
            if iv_rank > 80:
                signals.append(f"High IV rank ({iv_rank:.0f})")
                # High IV = sell premium
                if abs(current_price - gamma_data['nearest_strike']) / current_price < 0.02:
                    signals.append("Near strike with high IV")
                    strength -= 0.2  # Expect pinning
            elif iv_rank < 20:
                signals.append(f"Low IV rank ({iv_rank:.0f})")
                # Low IV = buy premium, expect movement
                strength += 0.1
            
            # 2. Put/Call Ratio Analysis
            pc_ratio = pc_data['pc_ratio']
            if pc_ratio > 1.5:
                signals.append(f"High P/C ratio ({pc_ratio:.2f}) - bearish sentiment")
                # Contrarian indicator
                if current_price < sma_20:
                    signals.append("Oversold with high put buying")
                    strength += 0.3
            elif pc_ratio < 0.7:
                signals.append(f"Low P/C ratio ({pc_ratio:.2f}) - bullish sentiment")
                if current_price > sma_20 * 1.05:
                    signals.append("Overbought with high call buying")
                    strength -= 0.3
            
            # 3. Gamma Exposure Analysis
            if gamma_data['pinning_detected']:
                signals.append(f"Pinning detected near ${gamma_data['nearest_strike']}")
                strength *= 0.5  # Reduce position size
                
                if gamma_data['near_expiration']:
                    signals.append("Near expiration - expect pin to break")
                    # After expiration, moves can be violent
                    if current_price > gamma_data['nearest_strike']:
                        strength += 0.2
                    else:
                        strength -= 0.2
            
            # 4. Volatility Term Structure
            if term_data['term_structure'] > 1.3:
                signals.append("Inverted vol term structure - stress")
                # Short-term stress often mean reverts
                if pc_ratio > 1.2:
                    strength += 0.2  # Buy the fear
            elif term_data['term_structure'] < 0.8:
                signals.append("Steep vol term structure - complacency")
                strength -= 0.1  # Caution on complacency
            
            # 5. Options Expiration Effects
            if gamma_data['days_to_friday'] == 0:  # Expiration day
                signals.append("Options expiration day")
                if gamma_data['distance_to_strike'] < 0.01:
                    signals.append("Expect volatile close")
                    strength *= 1.2  # Increase position for post-expiry move
            
            # 6. Volatility Regime
            current_vol = term_data['vol_5d']
            if current_vol > 0.3:  # 30% annualized
                signals.append("High volatility regime")
                # In high vol, be contrarian
                if pc_ratio > 1.0:
                    strength += 0.2
                else:
                    strength -= 0.2
            elif current_vol < 0.15:  # 15% annualized
                signals.append("Low volatility regime")
                # In low vol, follow trend
                if current_price > sma_20:
                    strength += 0.1
                else:
                    strength -= 0.1
            
            # 7. Skew Analysis
            if pc_data['skew'] > 1.3:
                signals.append("Negative skew - downside protection bid")
                if current_price < sma_20 * 0.98:
                    strength += 0.1  # Oversold with protection
            
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
                reason=f"Options Flow: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "iv_rank": float(iv_rank),
                    "estimated_iv": float(iv_data['estimated_iv']),
                    "pc_ratio": float(pc_ratio),
                    "nearest_strike": float(gamma_data['nearest_strike']),
                    "distance_to_strike": float(gamma_data['distance_to_strike']),
                    "pinning": gamma_data['pinning_detected'],
                    "days_to_expiry": gamma_data['days_to_friday'],
                    "vol_term_structure": float(term_data['term_structure']),
                    "signals": signals
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating Options Flow signal for {symbol}: {str(e)}")
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