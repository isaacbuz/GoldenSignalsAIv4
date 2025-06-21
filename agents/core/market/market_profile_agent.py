"""
Market Profile Agent
Analyzes time spent at price levels to identify market structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import yfinance as yf
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class MarketProfileAgent:
    """
    Market Profile trading agent
    
    Signals:
    - Initial Balance (IB) range
    - Range extension analysis
    - Time-based POC (Point of Control)
    - Market structure (trend/balance)
    """
    
    def __init__(self):
        self.name = "MarketProfileAgent"
        self.lookback_days = 10  # Days for profile analysis
        self.ib_hours = 1  # Initial balance period (first hour)
        self.time_bins = 24  # Time periods per day
        
    def analyze_daily_profile(self, daily_data: pd.Series) -> Dict[str, Any]:
        """Analyze single day's market profile"""
        open_price = daily_data['Open']
        high_price = daily_data['High']
        low_price = daily_data['Low']
        close_price = daily_data['Close']
        volume = daily_data['Volume']
        
        # Simulate Initial Balance (first 10% of day's range from open)
        ib_range = (high_price - low_price) * 0.3
        ib_high = open_price + ib_range / 2
        ib_low = open_price - ib_range / 2
        
        # Ensure IB is within day's range
        ib_high = min(ib_high, high_price)
        ib_low = max(ib_low, low_price)
        
        # Range extension analysis
        range_extension_up = high_price > ib_high
        range_extension_down = low_price < ib_low
        
        # Calculate time-based POC (simplified - assume normal distribution around VWAP)
        vwap = (high_price + low_price + close_price) / 3  # Simplified VWAP
        
        # Market structure
        day_range = high_price - low_price
        close_location = (close_price - low_price) / day_range if day_range > 0 else 0.5
        
        return {
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_high - ib_low,
            'range_extension_up': range_extension_up,
            'range_extension_down': range_extension_down,
            'tpoc': vwap,  # Time-based POC
            'close_location': close_location,
            'day_type': self._classify_day_type(close_location, range_extension_up, range_extension_down)
        }
    
    def _classify_day_type(self, close_location: float, ext_up: bool, ext_down: bool) -> str:
        """Classify the day type based on market profile"""
        if ext_up and ext_down:
            return "Neutral Day"  # Range extension both ways
        elif ext_up and close_location > 0.7:
            return "Trend Day Up"  # Strong uptrend
        elif ext_down and close_location < 0.3:
            return "Trend Day Down"  # Strong downtrend
        elif not ext_up and not ext_down:
            return "Balance Day"  # No range extension
        elif ext_up:
            return "Normal Variation Up"
        else:
            return "Normal Variation Down"
    
    def calculate_market_structure(self, profiles: List[Dict]) -> Dict[str, Any]:
        """Analyze market structure from multiple daily profiles"""
        # Count day types
        day_type_counts = defaultdict(int)
        for profile in profiles:
            day_type_counts[profile['day_type']] += 1
        
        # Determine market phase
        total_days = len(profiles)
        trend_days = day_type_counts.get('Trend Day Up', 0) + day_type_counts.get('Trend Day Down', 0)
        balance_days = day_type_counts.get('Balance Day', 0)
        
        if trend_days / total_days > 0.4:
            market_phase = "Trending"
        elif balance_days / total_days > 0.4:
            market_phase = "Balancing"
        else:
            market_phase = "Transitional"
        
        # Average metrics
        avg_ib_range = np.mean([p['ib_range'] for p in profiles])
        tpoc_trend = np.polyfit(range(len(profiles)), [p['tpoc'] for p in profiles], 1)[0]
        
        return {
            'market_phase': market_phase,
            'day_type_distribution': dict(day_type_counts),
            'avg_ib_range': avg_ib_range,
            'tpoc_trend': tpoc_trend,  # Positive = uptrend, negative = downtrend
        }
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Market Profile trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="1d")
            
            if df.empty or len(df) < self.lookback_days:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")
            
            # Analyze recent daily profiles
            recent_df = df.tail(self.lookback_days)
            profiles = []
            
            for idx, row in recent_df.iterrows():
                profile = self.analyze_daily_profile(row)
                profiles.append(profile)
            
            # Get today's data
            current_price = df['Close'].iloc[-1]
            today_profile = profiles[-1]
            yesterday_profile = profiles[-2] if len(profiles) > 1 else None
            
            # Analyze market structure
            market_structure = self.calculate_market_structure(profiles)
            
            signals = []
            strength = 0
            
            # 1. Market Phase Analysis
            market_phase = market_structure['market_phase']
            signals.append(f"Market phase: {market_phase}")
            
            if market_phase == "Trending":
                # Favor trend continuation
                if market_structure['tpoc_trend'] > 0:
                    signals.append("Uptrending market structure")
                    strength += 0.3
                else:
                    signals.append("Downtrending market structure")
                    strength -= 0.3
            elif market_phase == "Balancing":
                signals.append("Range-bound market")
                # Fade moves at extremes
                if today_profile['close_location'] > 0.8:
                    signals.append("Near range high - fade")
                    strength -= 0.2
                elif today_profile['close_location'] < 0.2:
                    signals.append("Near range low - fade")
                    strength += 0.2
            
            # 2. Initial Balance Analysis
            if current_price > today_profile['ib_high']:
                signals.append(f"Above IB high (${today_profile['ib_high']:.2f})")
                if today_profile['range_extension_up']:
                    signals.append("Confirmed range extension up")
                    strength += 0.3
            elif current_price < today_profile['ib_low']:
                signals.append(f"Below IB low (${today_profile['ib_low']:.2f})")
                if today_profile['range_extension_down']:
                    signals.append("Confirmed range extension down")
                    strength -= 0.3
            else:
                signals.append("Within Initial Balance")
                strength *= 0.7  # Reduce confidence within IB
            
            # 3. Day Type Analysis
            today_type = today_profile['day_type']
            if today_type == "Trend Day Up":
                signals.append("Trend day up pattern")
                strength += 0.4
            elif today_type == "Trend Day Down":
                signals.append("Trend day down pattern")
                strength -= 0.4
            elif today_type == "Balance Day":
                signals.append("Balance day - expect rotation")
                # Mean reversion likely
                if current_price > today_profile['tpoc']:
                    strength -= 0.1
                else:
                    strength += 0.1
            
            # 4. Compare to yesterday
            if yesterday_profile:
                # Check for acceptance above/below yesterday's range
                if current_price > yesterday_profile['ib_high'] * 1.01:
                    signals.append("Accepted above yesterday's IB")
                    strength += 0.2
                elif current_price < yesterday_profile['ib_low'] * 0.99:
                    signals.append("Accepted below yesterday's IB")
                    strength -= 0.2
                
                # IB range comparison
                ib_expansion = (today_profile['ib_range'] - yesterday_profile['ib_range']) / yesterday_profile['ib_range']
                if abs(ib_expansion) > 0.3:
                    if ib_expansion > 0:
                        signals.append("IB range expansion - volatility increasing")
                    else:
                        signals.append("IB range contraction - volatility decreasing")
            
            # 5. Volume analysis (simplified)
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = df['Volume'].tail(20).mean()
            
            if recent_volume > avg_volume * 1.2:
                signals.append("Above average volume")
                strength *= 1.1
            elif recent_volume < avg_volume * 0.8:
                signals.append("Below average volume")
                strength *= 0.9
            
            # Determine action
            if strength >= 0.35:
                action = "BUY"
            elif strength <= -0.35:
                action = "SELL"
            else:
                action = "NEUTRAL"
            
            confidence = min(abs(strength), 1.0)
            
            return self._create_signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=f"Market Profile: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "ib_high": float(today_profile['ib_high']),
                    "ib_low": float(today_profile['ib_low']),
                    "tpoc": float(today_profile['tpoc']),
                    "day_type": today_profile['day_type'],
                    "market_phase": market_phase,
                    "close_location": float(today_profile['close_location']),
                    "signals": signals
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating Market Profile signal for {symbol}: {str(e)}")
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