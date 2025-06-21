"""
Order Flow Agent
Analyzes order flow patterns and market microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OrderFlowAgent:
    """
    Order Flow trading agent
    
    Signals:
    - Buy/sell pressure imbalances
    - Absorption patterns
    - Aggressive vs passive flow
    - Large order detection
    """
    
    def __init__(self):
        self.name = "OrderFlowAgent"
        self.lookback_periods = 20
        self.volume_threshold = 1.5  # Multiplier for unusual volume
        
    def estimate_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate order flow from price and volume data"""
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Estimate buy/sell volume (simplified)
        # Positive returns = more buy volume, negative = more sell volume
        df['buy_volume'] = df['Volume'] * (df['returns'] > 0).astype(int)
        df['sell_volume'] = df['Volume'] * (df['returns'] < 0).astype(int)
        
        # Adjust based on price action within candle
        # If close near high, more buying pressure
        df['buying_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['buying_pressure'] = df['buying_pressure'].fillna(0.5)
        
        # Refine buy/sell estimates
        df['est_buy_volume'] = df['Volume'] * df['buying_pressure']
        df['est_sell_volume'] = df['Volume'] * (1 - df['buying_pressure'])
        
        # Delta (buy - sell volume)
        df['delta'] = df['est_buy_volume'] - df['est_sell_volume']
        df['cumulative_delta'] = df['delta'].cumsum()
        
        # Volume-weighted average price moves
        df['vwap_move'] = df['returns'] * df['Volume']
        
        # Detect absorption (high volume, small price move)
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['absorption'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5) & \
                          (df['price_range'] < df['price_range'].rolling(20).mean() * 0.7)
        
        return df
    
    def detect_large_orders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential large order activity"""
        large_orders = []
        
        # Volume spikes
        volume_mean = df['Volume'].rolling(20).mean()
        volume_std = df['Volume'].rolling(20).std()
        
        for i in range(len(df)):
            if i < 20:
                continue
                
            current_volume = df['Volume'].iloc[i]
            threshold = volume_mean.iloc[i] + 2 * volume_std.iloc[i]
            
            if current_volume > threshold:
                # Determine order type based on price action
                price_change = df['Close'].iloc[i] - df['Open'].iloc[i]
                order_type = "Large Buy" if price_change > 0 else "Large Sell"
                
                large_orders.append({
                    'index': i,
                    'type': order_type,
                    'volume': current_volume,
                    'price_impact': abs(price_change) / df['Open'].iloc[i] * 100
                })
        
        return large_orders
    
    def analyze_flow_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow patterns"""
        recent_df = df.tail(self.lookback_periods)
        
        # Calculate flow metrics
        total_buy_volume = recent_df['est_buy_volume'].sum()
        total_sell_volume = recent_df['est_sell_volume'].sum()
        buy_sell_ratio = total_buy_volume / total_sell_volume if total_sell_volume > 0 else 1
        
        # Delta analysis
        recent_delta = recent_df['delta'].sum()
        delta_trend = np.polyfit(range(len(recent_df)), recent_df['cumulative_delta'].values, 1)[0]
        
        # Absorption patterns
        absorption_count = recent_df['absorption'].sum()
        
        # Momentum
        volume_momentum = recent_df['Volume'].tail(5).mean() / recent_df['Volume'].tail(20).mean()
        
        return {
            'buy_sell_ratio': buy_sell_ratio,
            'recent_delta': recent_delta,
            'delta_trend': delta_trend,
            'absorption_count': absorption_count,
            'volume_momentum': volume_momentum,
            'dominant_side': 'buy' if buy_sell_ratio > 1.1 else 'sell' if buy_sell_ratio < 0.9 else 'neutral'
        }
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Order Flow trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2mo", interval="1d")
            
            if df.empty or len(df) < 30:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")
            
            # Estimate order flow
            df = self.estimate_order_flow(df)
            
            # Analyze patterns
            flow_patterns = self.analyze_flow_patterns(df)
            large_orders = self.detect_large_orders(df.tail(10))  # Recent large orders
            
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(20).mean()
            
            signals = []
            strength = 0
            
            # 1. Buy/Sell Pressure Analysis
            buy_sell_ratio = flow_patterns['buy_sell_ratio']
            if buy_sell_ratio > 1.3:
                signals.append(f"Strong buying pressure (ratio: {buy_sell_ratio:.2f})")
                strength += 0.4
            elif buy_sell_ratio < 0.7:
                signals.append(f"Strong selling pressure (ratio: {buy_sell_ratio:.2f})")
                strength -= 0.4
            elif buy_sell_ratio > 1.1:
                signals.append(f"Moderate buying pressure (ratio: {buy_sell_ratio:.2f})")
                strength += 0.2
            elif buy_sell_ratio < 0.9:
                signals.append(f"Moderate selling pressure (ratio: {buy_sell_ratio:.2f})")
                strength -= 0.2
            else:
                signals.append("Balanced order flow")
            
            # 2. Delta Analysis
            if flow_patterns['delta_trend'] > 0:
                signals.append("Positive delta trend (accumulation)")
                strength += 0.2
            elif flow_patterns['delta_trend'] < 0:
                signals.append("Negative delta trend (distribution)")
                strength -= 0.2
            
            # 3. Large Order Detection
            recent_large_buys = sum(1 for order in large_orders if order['type'] == "Large Buy")
            recent_large_sells = sum(1 for order in large_orders if order['type'] == "Large Sell")
            
            if recent_large_buys > recent_large_sells:
                signals.append(f"Large buy orders detected ({recent_large_buys})")
                strength += 0.3
            elif recent_large_sells > recent_large_buys:
                signals.append(f"Large sell orders detected ({recent_large_sells})")
                strength -= 0.3
            
            # 4. Absorption Analysis
            if flow_patterns['absorption_count'] > 2:
                # High absorption often precedes reversals
                signals.append(f"High absorption detected ({flow_patterns['absorption_count']} days)")
                
                # Check direction of absorption
                recent_absorption = df[df['absorption']].tail(3)
                if not recent_absorption.empty:
                    avg_delta = recent_absorption['delta'].mean()
                    if avg_delta > 0:
                        signals.append("Buy absorption - potential support")
                        strength += 0.2
                    else:
                        signals.append("Sell absorption - potential resistance")
                        strength -= 0.2
            
            # 5. Volume Flow Analysis
            if current_volume > avg_volume * self.volume_threshold:
                # High volume with price direction
                price_change = df['returns'].iloc[-1]
                if price_change > 0.01:
                    signals.append("High volume buying")
                    strength += 0.2
                elif price_change < -0.01:
                    signals.append("High volume selling")
                    strength -= 0.2
                else:
                    signals.append("High volume equilibrium")
            
            # 6. Flow Momentum
            if flow_patterns['volume_momentum'] > 1.2:
                signals.append("Increasing volume momentum")
                strength *= 1.1
            elif flow_patterns['volume_momentum'] < 0.8:
                signals.append("Decreasing volume momentum")
                strength *= 0.9
            
            # 7. Price Level Analysis
            # Check if breaking key levels with volume
            sma_20 = df['Close'].tail(20).mean()
            if current_price > sma_20 * 1.01 and flow_patterns['dominant_side'] == 'buy':
                signals.append("Breaking above MA with buy flow")
                strength += 0.1
            elif current_price < sma_20 * 0.99 and flow_patterns['dominant_side'] == 'sell':
                signals.append("Breaking below MA with sell flow")
                strength -= 0.1
            
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
                reason=f"Order Flow: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "buy_sell_ratio": float(buy_sell_ratio),
                    "delta_trend": float(flow_patterns['delta_trend']),
                    "large_buy_orders": recent_large_buys,
                    "large_sell_orders": recent_large_sells,
                    "absorption_days": flow_patterns['absorption_count'],
                    "volume_momentum": float(flow_patterns['volume_momentum']),
                    "dominant_flow": flow_patterns['dominant_side'],
                    "signals": signals
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating Order Flow signal for {symbol}: {str(e)}")
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