#!/usr/bin/env python3
"""
🎯 GoldenSignalsAI - Precise Options Signal Generator
Ultra-specific signals with exact entry/exit, strikes, and risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from dataclasses import dataclass

@dataclass
class PreciseOptionsSignal:
    """Detailed options trade signal with all specifics"""
    # Trade Identification
    symbol: str
    signal_type: str  # BUY_CALL or BUY_PUT
    confidence: float
    
    # Timing - PRECISE
    entry_time: str  # "Market Open", "10:30 AM ET", "3:45 PM ET"
    entry_date: str  # Today, Tomorrow
    hold_duration: str  # "2-3 days", "Until Friday close"
    exit_time: str  # "Before 3:00 PM ET Friday"
    
    # Price Levels - EXACT
    current_price: float
    entry_trigger: float  # Exact price to enter
    entry_range: Tuple[float, float]  # Acceptable entry range
    
    # Options Specifics
    strike_price: float  # Recommended strike
    expiration: str  # Specific expiration date
    option_type: str  # "Weekly", "Monthly"
    
    # Risk Management - PRECISE
    stop_loss: float  # Based on underlying price
    stop_loss_pct: float  # As percentage
    take_profit_1: float  # First target (50% position)
    take_profit_2: float  # Second target (remaining 50%)
    max_risk_per_trade: float  # Dollar amount
    
    # Exit Conditions
    exit_conditions: List[str]  # Multiple exit triggers
    time_stop: str  # "Exit by 3:00 PM if no movement"
    
    # Reasoning
    technical_setup: str
    key_levels: Dict[str, float]
    risk_factors: List[str]
    
class PreciseSignalGenerator:
    """Generate ultra-specific options trading signals"""
    
    def __init__(self):
        self.market_hours = {
            'open': '09:30',
            'close': '16:00',
            'power_hour': '15:00',
            'last_30': '15:30'
        }
        
    def generate_precise_signal(self, symbol: str, data: pd.DataFrame) -> PreciseOptionsSignal:
        """Generate precise options trading signal"""
        
        # Analyze current setup
        setup = self.analyze_setup(data)
        
        # Determine signal type
        signal_type = self.determine_signal_type(setup)
        
        if signal_type == "HOLD":
            return None
        
        # Calculate all precise levels
        current_price = setup['current_price']
        
        # Entry timing based on patterns
        entry_timing = self.calculate_entry_timing(setup)
        
        # Options specifics
        options_details = self.calculate_options_details(
            current_price, 
            signal_type, 
            setup['volatility']
        )
        
        # Risk management levels
        risk_levels = self.calculate_risk_levels(
            current_price,
            signal_type,
            setup
        )
        
        # Exit conditions
        exit_conditions = self.determine_exit_conditions(
            signal_type,
            setup,
            options_details['expiration_date']
        )
        
        return PreciseOptionsSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=setup['confidence'],
            
            # Precise timing
            entry_time=entry_timing['time'],
            entry_date=entry_timing['date'],
            hold_duration=entry_timing['duration'],
            exit_time=entry_timing['exit_by'],
            
            # Exact levels
            current_price=current_price,
            entry_trigger=risk_levels['entry_trigger'],
            entry_range=(risk_levels['entry_low'], risk_levels['entry_high']),
            
            # Options details
            strike_price=options_details['strike'],
            expiration=options_details['expiration_date'],
            option_type=options_details['type'],
            
            # Risk management
            stop_loss=risk_levels['stop_loss'],
            stop_loss_pct=risk_levels['stop_loss_pct'],
            take_profit_1=risk_levels['target_1'],
            take_profit_2=risk_levels['target_2'],
            max_risk_per_trade=risk_levels['max_risk'],
            
            # Exit rules
            exit_conditions=exit_conditions['conditions'],
            time_stop=exit_conditions['time_stop'],
            
            # Analysis
            technical_setup=setup['pattern'],
            key_levels=setup['key_levels'],
            risk_factors=setup['risks']
        )
    
    def analyze_setup(self, data: pd.DataFrame) -> Dict:
        """Analyze current technical setup"""
        latest = data.iloc[-1]
        
        # Key calculations
        atr = self.calculate_atr(data)
        support = data['Low'].rolling(20).min().iloc[-1]
        resistance = data['High'].rolling(20).max().iloc[-1]
        
        # Pattern detection
        pattern = self.detect_pattern(data)
        
        # Momentum analysis
        rsi = self.calculate_rsi(data)
        macd_signal = self.get_macd_signal(data)
        
        # Volume analysis
        volume_spike = latest['Volume'] > data['Volume'].rolling(20).mean().iloc[-1] * 1.5
        
        # Trend strength
        trend_strength = self.calculate_trend_strength(data)
        
        # Calculate confidence
        confidence = self.calculate_setup_confidence(
            pattern, rsi, macd_signal, volume_spike, trend_strength
        )
        
        return {
            'current_price': latest['Close'],
            'atr': atr,
            'volatility': data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252),
            'pattern': pattern,
            'key_levels': {
                'support': support,
                'resistance': resistance,
                'pivot': (latest['High'] + latest['Low'] + latest['Close']) / 3,
                'r1': 2 * ((latest['High'] + latest['Low'] + latest['Close']) / 3) - latest['Low'],
                's1': 2 * ((latest['High'] + latest['Low'] + latest['Close']) / 3) - latest['High']
            },
            'rsi': rsi,
            'macd_signal': macd_signal,
            'volume_spike': volume_spike,
            'trend_strength': trend_strength,
            'confidence': confidence,
            'risks': self.identify_risks(data)
        }
    
    def determine_signal_type(self, setup: Dict) -> str:
        """Determine specific signal type based on setup"""
        
        # Strong bullish signals
        if (setup['rsi'] < 30 and 
            setup['pattern'] in ['double_bottom', 'bullish_flag', 'ascending_triangle'] and
            setup['macd_signal'] == 'bullish_crossover' and
            setup['confidence'] > 0.7):
            return "BUY_CALL"
        
        # Strong bearish signals
        elif (setup['rsi'] > 70 and
              setup['pattern'] in ['double_top', 'bearish_flag', 'descending_triangle'] and
              setup['macd_signal'] == 'bearish_crossover' and
              setup['confidence'] > 0.7):
            return "BUY_PUT"
        
        # Breakout signals
        elif setup['current_price'] > setup['key_levels']['resistance'] and setup['volume_spike']:
            return "BUY_CALL"
        
        elif setup['current_price'] < setup['key_levels']['support'] and setup['volume_spike']:
            return "BUY_PUT"
        
        else:
            return "HOLD"
    
    def calculate_entry_timing(self, setup: Dict) -> Dict:
        """Calculate precise entry timing"""
        now = datetime.now()
        
        # Best entry times based on market dynamics
        if setup['pattern'] in ['gap_up', 'gap_down']:
            entry_time = "Wait for 10:00 AM ET (30 min after open)"
            entry_date = "Today"
        elif setup['rsi'] < 30:  # Oversold bounce
            entry_time = "Market Open (9:30 AM ET)"
            entry_date = "Tomorrow" if now.hour > 15 else "Today"
        elif setup['pattern'] == 'breakout':
            entry_time = "On breakout confirmation with volume"
            entry_date = "When trigger hits"
        else:
            entry_time = "10:30 AM - 11:00 AM ET (after morning volatility)"
            entry_date = "Next trading day"
        
        # Hold duration based on signal strength
        if setup['confidence'] > 0.8:
            duration = "2-3 days"
            exit_by = "Thursday 3:00 PM ET"
        else:
            duration = "1-2 days"
            exit_by = "Wednesday close"
        
        return {
            'time': entry_time,
            'date': entry_date,
            'duration': duration,
            'exit_by': exit_by
        }
    
    def calculate_options_details(self, current_price: float, signal_type: str, volatility: float) -> Dict:
        """Calculate specific options contract details"""
        
        # Days to expiration based on volatility
        if volatility > 0.4:  # High volatility
            dte = 7  # Weekly options
            option_type = "Weekly"
        else:
            dte = 30  # Monthly options
            option_type = "Monthly"
        
        # Strike selection
        if signal_type == "BUY_CALL":
            # For calls: slightly OTM for better leverage
            if volatility > 0.3:
                strike = self.round_to_strike(current_price * 1.01)  # 1% OTM
            else:
                strike = self.round_to_strike(current_price * 1.005)  # 0.5% OTM
        else:  # BUY_PUT
            if volatility > 0.3:
                strike = self.round_to_strike(current_price * 0.99)  # 1% OTM
            else:
                strike = self.round_to_strike(current_price * 0.995)  # 0.5% OTM
        
        # Expiration date
        expiration = self.get_next_expiration(dte)
        
        return {
            'strike': strike,
            'expiration_date': expiration,
            'type': option_type,
            'dte': dte
        }
    
    def calculate_risk_levels(self, current_price: float, signal_type: str, setup: Dict) -> Dict:
        """Calculate precise risk management levels"""
        
        atr = setup['atr']
        
        if signal_type == "BUY_CALL":
            # Entry trigger (breakout or bounce level)
            if setup['pattern'] == 'breakout':
                entry_trigger = setup['key_levels']['resistance'] + 0.10
            else:
                entry_trigger = current_price + 0.05
            
            # Entry range
            entry_low = entry_trigger - (atr * 0.2)
            entry_high = entry_trigger + (atr * 0.2)
            
            # Stop loss below recent support or 1 ATR
            stop_loss = max(setup['key_levels']['support'], current_price - atr)
            
            # Targets based on R:R and resistance
            target_1 = current_price + (atr * 1.5)  # 1.5:1 R:R
            target_2 = min(current_price + (atr * 3), setup['key_levels']['r1'])  # 3:1 or resistance
            
        else:  # BUY_PUT
            # Entry trigger
            if setup['pattern'] == 'breakdown':
                entry_trigger = setup['key_levels']['support'] - 0.10
            else:
                entry_trigger = current_price - 0.05
            
            # Entry range
            entry_low = entry_trigger - (atr * 0.2)
            entry_high = entry_trigger + (atr * 0.2)
            
            # Stop loss above recent resistance or 1 ATR
            stop_loss = min(setup['key_levels']['resistance'], current_price + atr)
            
            # Targets
            target_1 = current_price - (atr * 1.5)
            target_2 = max(current_price - (atr * 3), setup['key_levels']['s1'])
        
        # Calculate percentages
        stop_loss_pct = abs(stop_loss - current_price) / current_price * 100
        
        # Max risk per trade (example: $500)
        max_risk = 500
        
        return {
            'entry_trigger': round(entry_trigger, 2),
            'entry_low': round(entry_low, 2),
            'entry_high': round(entry_high, 2),
            'stop_loss': round(stop_loss, 2),
            'stop_loss_pct': round(stop_loss_pct, 1),
            'target_1': round(target_1, 2),
            'target_2': round(target_2, 2),
            'max_risk': max_risk
        }
    
    def determine_exit_conditions(self, signal_type: str, setup: Dict, expiration: str) -> Dict:
        """Determine multiple exit conditions"""
        
        conditions = []
        
        # Profit targets
        conditions.append("Take 50% profit at Target 1")
        conditions.append("Take remaining 50% at Target 2")
        
        # Stop loss
        conditions.append(f"Exit if underlying hits stop loss")
        
        # Time-based exits
        conditions.append("Exit if no movement after 2 days")
        conditions.append(f"Exit all positions by {expiration} - 1 day")
        
        # Technical exits
        if signal_type == "BUY_CALL":
            conditions.append("Exit if RSI > 70 (overbought)")
            conditions.append("Exit on bearish MACD crossover")
        else:
            conditions.append("Exit if RSI < 30 (oversold)")
            conditions.append("Exit on bullish MACD crossover")
        
        # Time stop
        time_stop = "Exit by 3:00 PM ET if position is flat or negative on day 2"
        
        return {
            'conditions': conditions,
            'time_stop': time_stop
        }
    
    # Helper methods
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean().iloc[-1]
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    def get_macd_signal(self, data: pd.DataFrame) -> str:
        macd = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return "bullish_crossover"
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return "bearish_crossover"
        else:
            return "neutral"
    
    def detect_pattern(self, data: pd.DataFrame) -> str:
        # Simplified pattern detection
        closes = data['Close'].tail(10)
        
        if closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3]:
            return "uptrend"
        elif closes.iloc[-1] < closes.iloc[-2] < closes.iloc[-3]:
            return "downtrend"
        else:
            return "consolidation"
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        sma20 = data['Close'].rolling(20).mean()
        sma50 = data['Close'].rolling(50).mean()
        
        if data['Close'].iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]:
            return 0.8
        elif data['Close'].iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1]:
            return 0.2
        else:
            return 0.5
    
    def calculate_setup_confidence(self, pattern: str, rsi: float, macd: str, volume: bool, trend: float) -> float:
        confidence = 0.5
        
        # Pattern bonus
        if pattern in ['double_bottom', 'double_top', 'breakout']:
            confidence += 0.2
        
        # RSI extremes
        if rsi < 30 or rsi > 70:
            confidence += 0.1
        
        # MACD confirmation
        if macd in ['bullish_crossover', 'bearish_crossover']:
            confidence += 0.1
        
        # Volume confirmation
        if volume:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def identify_risks(self, data: pd.DataFrame) -> List[str]:
        risks = []
        
        # Volatility risk
        vol = data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        if vol > 0.4:
            risks.append("High volatility - use wider stops")
        
        # Trend risk
        sma50 = data['Close'].rolling(50).mean().iloc[-1]
        if abs(data['Close'].iloc[-1] - sma50) / sma50 > 0.05:
            risks.append("Extended from 50-day average")
        
        # Time decay risk
        if datetime.now().weekday() >= 3:  # Thursday or Friday
            risks.append("Weekend time decay risk for weekly options")
        
        return risks
    
    def round_to_strike(self, price: float) -> float:
        """Round to nearest standard strike price"""
        if price < 50:
            return round(price)  # $1 strikes
        elif price < 200:
            return round(price / 5) * 5  # $5 strikes
        else:
            return round(price / 10) * 10  # $10 strikes
    
    def get_next_expiration(self, dte: int) -> str:
        """Get next options expiration date"""
        target = datetime.now() + timedelta(days=dte)
        
        # Find next Friday
        days_until_friday = (4 - target.weekday()) % 7
        if days_until_friday == 0 and target.hour > 16:
            days_until_friday = 7
        
        expiration = target + timedelta(days=days_until_friday)
        return expiration.strftime("%Y-%m-%d")

def demonstrate_precise_signals():
    """Show precise signal examples"""
    generator = PreciseSignalGenerator()
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Bullish scenario
    bull_data = pd.DataFrame({
        'Close': 150 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))),
        'Open': 150 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)) + np.random.normal(0, 0.005, 100)),
        'High': 152 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))),
        'Low': 148 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))),
        'Volume': np.random.lognormal(17, 0.5, 100)
    }, index=dates)
    
    # Generate signal
    signal = generator.generate_precise_signal('AAPL', bull_data)
    
    if signal:
        print("="*70)
        print("🎯 PRECISE OPTIONS SIGNAL - AAPL")
        print("="*70)
        
        # Signal header
        emoji = "🟢" if signal.signal_type == "BUY_CALL" else "🔴"
        print(f"\n{emoji} {signal.signal_type}")
        print(f"🎯 Confidence: {signal.confidence:.1%}")
        
        # Timing - EXACT
        print(f"\n⏰ ENTRY TIMING:")
        print(f"   Date: {signal.entry_date}")
        print(f"   Time: {signal.entry_time}")
        print(f"   Hold: {signal.hold_duration}")
        print(f"   Exit by: {signal.exit_time}")
        
        # Options contract - SPECIFIC
        print(f"\n📄 OPTIONS CONTRACT:")
        print(f"   Strike: ${signal.strike_price}")
        print(f"   Expiration: {signal.expiration}")
        print(f"   Type: {signal.option_type}")
        
        # Entry levels - PRECISE
        print(f"\n📍 ENTRY LEVELS:")
        print(f"   Current: ${signal.current_price:.2f}")
        print(f"   Entry Trigger: ${signal.entry_trigger:.2f}")
        print(f"   Entry Range: ${signal.entry_range[0]:.2f} - ${signal.entry_range[1]:.2f}")
        
        # Risk management - EXACT
        print(f"\n🛡️ RISK MANAGEMENT:")
        print(f"   Stop Loss: ${signal.stop_loss:.2f} ({signal.stop_loss_pct:.1f}%)")
        print(f"   Target 1 (50%): ${signal.take_profit_1:.2f}")
        print(f"   Target 2 (50%): ${signal.take_profit_2:.2f}")
        print(f"   Max Risk: ${signal.max_risk_per_trade}")
        
        # Exit rules - CLEAR
        print(f"\n🚪 EXIT CONDITIONS:")
        for i, condition in enumerate(signal.exit_conditions, 1):
            print(f"   {i}. {condition}")
        print(f"   ⏱️ {signal.time_stop}")
        
        # Key levels
        print(f"\n📊 KEY LEVELS:")
        for level, price in signal.key_levels.items():
            print(f"   {level.capitalize()}: ${price:.2f}")
        
        # Risks
        if signal.risk_factors:
            print(f"\n⚠️ RISK FACTORS:")
            for risk in signal.risk_factors:
                print(f"   • {risk}")
        
        # Trade summary
        print(f"\n💎 TRADE SUMMARY:")
        print(f"   Setup: {signal.technical_setup}")
        r_r_ratio = (signal.take_profit_1 - signal.entry_trigger) / (signal.entry_trigger - signal.stop_loss)
        print(f"   Risk/Reward: {abs(r_r_ratio):.1f}:1")
        
        print("\n" + "="*70)
        print("⚡ QUICK EXECUTION GUIDE:")
        print(f"1. Set alert for ${signal.entry_trigger:.2f}")
        print(f"2. Buy {signal.option_type} {signal.strike_price} {signal.signal_type.split('_')[1]} expiring {signal.expiration}")
        print(f"3. Set stop loss at underlying price ${signal.stop_loss:.2f}")
        print(f"4. Take profits at ${signal.take_profit_1:.2f} and ${signal.take_profit_2:.2f}")
        print("="*70)

if __name__ == "__main__":
    demonstrate_precise_signals() 