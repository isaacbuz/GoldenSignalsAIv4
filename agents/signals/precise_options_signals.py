"""
🎯 Precise Options Signal Agent
Generates ultra-specific trading signals with exact entries, exits, and risk management
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yfinance as yf

@dataclass
class PreciseOptionsSignal:
    """Complete options trading signal with all specifics"""
    # Identification
    symbol: str
    signal_id: str
    generated_at: datetime
    
    # Trade Direction
    signal_type: str  # BUY_CALL or BUY_PUT
    confidence: float  # 0-100
    priority: str  # HIGH, MEDIUM, LOW
    
    # Precise Timing
    entry_window: Dict[str, str]  # date, start_time, end_time
    hold_duration: str  # "2-3 days", "1 week", etc.
    expiration_warning: str  # When to exit regardless
    
    # Options Contract
    strike_price: float
    expiration_date: str
    contract_type: str  # Weekly, Monthly
    max_premium: float  # Don't pay more than this
    
    # Entry Levels
    current_price: float
    entry_trigger: float  # Exact price to enter
    entry_zone: Tuple[float, float]  # Acceptable range
    
    # Risk Management
    stop_loss: float  # Based on underlying
    stop_loss_pct: float
    position_size: int  # Number of contracts
    max_risk_dollars: float
    
    # Profit Targets
    targets: List[Dict[str, float]]  # price, percentage to exit
    risk_reward_ratio: float
    
    # Exit Conditions
    exit_rules: List[str]
    time_based_exits: Dict[str, str]
    
    # Technical Justification
    setup_name: str
    key_indicators: Dict[str, float]
    chart_patterns: List[str]
    
    # Action Items
    alerts_to_set: List[str]
    pre_entry_checklist: List[str]

class PreciseSignalGenerator:
    """Generate precise, actionable options signals"""
    
    def __init__(self):
        self.indicators_config = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2
        }
        
        # Market session times (ET)
        self.market_sessions = {
            'pre_market': ('04:00', '09:30'),
            'open_volatile': ('09:30', '10:00'),
            'morning_trend': ('10:00', '11:30'),
            'lunch_chop': ('11:30', '13:00'),
            'afternoon_trend': ('13:00', '15:00'),
            'power_hour': ('15:00', '16:00'),
            'after_hours': ('16:00', '20:00')
        }
        
    def analyze_symbol(self, symbol: str) -> Optional[PreciseOptionsSignal]:
        """Analyze symbol and generate precise signal if setup exists"""
        
        # Get market data
        data = self._fetch_market_data(symbol)
        if data is None or len(data) < 50:
            return None
        
        # Calculate all indicators
        indicators = self._calculate_indicators(data)
        
        # Detect patterns and setups
        setup = self._identify_setup(data, indicators)
        if not setup:
            return None
        
        # Generate precise signal
        signal = self._generate_signal(symbol, data, indicators, setup)
        
        return signal
    
    def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch recent market data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get daily data for pattern analysis
            daily_data = ticker.history(period='3mo', interval='1d')
            
            # Get intraday data for precise timing
            intraday_data = ticker.history(period='5d', interval='15m')
            
            if daily_data.empty:
                return None
            
            return daily_data
        except:
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # Trend Indicators
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_9 = close.ewm(span=9).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        # ATR for stop loss calculation
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        
        # Bollinger Bands
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Support and Resistance
        support = low.rolling(20).min()
        resistance = high.rolling(20).max()
        
        # Volume indicators
        volume_sma = volume.rolling(20).mean()
        volume_ratio = volume / volume_sma
        
        return {
            'current_price': close.iloc[-1],
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1],
            'ema_9': ema_9.iloc[-1],
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2],
            'macd_line': macd_line.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'macd_histogram': macd_histogram.iloc[-1],
            'macd_cross': self._detect_macd_cross(macd_line, macd_signal),
            'atr': atr.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'bb_position': (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]),
            'support': support.iloc[-1],
            'resistance': resistance.iloc[-1],
            'volume_ratio': volume_ratio.iloc[-1],
            'trend_strength': self._calculate_trend_strength(close, sma_20, sma_50)
        }
    
    def _identify_setup(self, data: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Identify high-probability setup"""
        
        setups = []
        
        # Oversold Bounce Setup
        if indicators['rsi'] < 30 and indicators['rsi'] > indicators['rsi_prev']:
            setups.append({
                'name': 'Oversold Bounce',
                'type': 'BUY_CALL',
                'confidence': 75,
                'entry_timing': 'morning_trend',
                'hold_days': 2
            })
        
        # Overbought Reversal
        if indicators['rsi'] > 70 and indicators['rsi'] < indicators['rsi_prev']:
            setups.append({
                'name': 'Overbought Reversal',
                'type': 'BUY_PUT',
                'confidence': 75,
                'entry_timing': 'morning_trend',
                'hold_days': 2
            })
        
        # MACD Crossover
        if indicators['macd_cross'] == 'bullish':
            setups.append({
                'name': 'MACD Bullish Cross',
                'type': 'BUY_CALL',
                'confidence': 70,
                'entry_timing': 'open_volatile',
                'hold_days': 3
            })
        elif indicators['macd_cross'] == 'bearish':
            setups.append({
                'name': 'MACD Bearish Cross',
                'type': 'BUY_PUT',
                'confidence': 70,
                'entry_timing': 'open_volatile',
                'hold_days': 3
            })
        
        # Bollinger Band Squeeze Breakout
        if indicators['bb_position'] > 1.0 and indicators['volume_ratio'] > 1.5:
            setups.append({
                'name': 'BB Breakout Up',
                'type': 'BUY_CALL',
                'confidence': 80,
                'entry_timing': 'immediate',
                'hold_days': 1
            })
        elif indicators['bb_position'] < 0.0 and indicators['volume_ratio'] > 1.5:
            setups.append({
                'name': 'BB Breakdown',
                'type': 'BUY_PUT',
                'confidence': 80,
                'entry_timing': 'immediate',
                'hold_days': 1
            })
        
        # Support/Resistance Break
        if indicators['current_price'] > indicators['resistance'] * 1.001:
            setups.append({
                'name': 'Resistance Break',
                'type': 'BUY_CALL',
                'confidence': 85,
                'entry_timing': 'immediate',
                'hold_days': 2
            })
        elif indicators['current_price'] < indicators['support'] * 0.999:
            setups.append({
                'name': 'Support Break',
                'type': 'BUY_PUT',
                'confidence': 85,
                'entry_timing': 'immediate',
                'hold_days': 2
            })
        
        # Return highest confidence setup
        if setups:
            return max(setups, key=lambda x: x['confidence'])
        return None
    
    def _generate_signal(self, symbol: str, data: pd.DataFrame, 
                        indicators: Dict, setup: Dict) -> PreciseOptionsSignal:
        """Generate precise, actionable signal"""
        
        current_price = indicators['current_price']
        atr = indicators['atr']
        
        # Calculate entry levels
        if setup['type'] == 'BUY_CALL':
            entry_trigger = current_price * 1.001  # 0.1% above current
            entry_zone = (current_price * 0.999, current_price * 1.003)
            stop_loss = max(indicators['support'], current_price - (atr * 1.5))
            target_1 = current_price + (atr * 2)
            target_2 = current_price + (atr * 3.5)
        else:  # BUY_PUT
            entry_trigger = current_price * 0.999  # 0.1% below current
            entry_zone = (current_price * 0.997, current_price * 1.001)
            stop_loss = min(indicators['resistance'], current_price + (atr * 1.5))
            target_1 = current_price - (atr * 2)
            target_2 = current_price - (atr * 3.5)
        
        # Calculate options details
        strike = self._select_strike(current_price, setup['type'])
        expiration = self._select_expiration(setup['hold_days'])
        
        # Position sizing
        risk_per_trade = 500  # $500 max risk
        stop_distance = abs(current_price - stop_loss)
        stop_pct = (stop_distance / current_price) * 100
        
        # Determine entry timing
        entry_window = self._determine_entry_window(setup['entry_timing'])
        
        # Generate alerts
        alerts = [
            f"Price alert: {symbol} crosses ${entry_trigger:.2f}",
            f"Price alert: {symbol} hits ${stop_loss:.2f} (STOP LOSS)",
            f"Price alert: {symbol} hits ${target_1:.2f} (Target 1)",
            f"Price alert: {symbol} hits ${target_2:.2f} (Target 2)"
        ]
        
        # Exit rules
        exit_rules = [
            f"Exit 50% at ${target_1:.2f} (Target 1)",
            f"Exit remaining 50% at ${target_2:.2f} (Target 2)",
            f"Hard stop if {symbol} crosses ${stop_loss:.2f}",
            f"Time stop: Exit all positions by {expiration} minus 1 day",
            "Exit if daily RSI crosses above 70" if setup['type'] == 'BUY_CALL' else "Exit if daily RSI crosses below 30",
            "Exit on opposite MACD crossover signal"
        ]
        
        # Risk/Reward calculation
        risk = abs(entry_trigger - stop_loss)
        reward = target_1 - entry_trigger if setup['type'] == 'BUY_CALL' else entry_trigger - target_1
        risk_reward = reward / risk if risk > 0 else 0
        
        return PreciseOptionsSignal(
            symbol=symbol,
            signal_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            
            signal_type=setup['type'],
            confidence=setup['confidence'],
            priority="HIGH" if setup['confidence'] > 80 else "MEDIUM",
            
            entry_window=entry_window,
            hold_duration=f"{setup['hold_days']}-{setup['hold_days']+1} days",
            expiration_warning=f"Must exit by {expiration} minus 1 day",
            
            strike_price=strike,
            expiration_date=expiration,
            contract_type="Weekly" if setup['hold_days'] <= 5 else "Monthly",
            max_premium=3.00,  # Don't pay more than $3
            
            current_price=current_price,
            entry_trigger=round(entry_trigger, 2),
            entry_zone=(round(entry_zone[0], 2), round(entry_zone[1], 2)),
            
            stop_loss=round(stop_loss, 2),
            stop_loss_pct=round(stop_pct, 1),
            position_size=2,  # Start with 2 contracts
            max_risk_dollars=risk_per_trade,
            
            targets=[
                {'price': round(target_1, 2), 'exit_pct': 50},
                {'price': round(target_2, 2), 'exit_pct': 50}
            ],
            risk_reward_ratio=round(risk_reward, 1),
            
            exit_rules=exit_rules,
            time_based_exits={
                'intraday': "Exit if no movement by 2:00 PM on entry day",
                'multi_day': f"Reduce position by 50% after {setup['hold_days']} days",
                'expiration': f"Exit all by {expiration} minus 1 day"
            },
            
            setup_name=setup['name'],
            key_indicators={
                'RSI': round(indicators['rsi'], 1),
                'MACD': indicators['macd_cross'],
                'ATR': round(atr, 2),
                'Volume': f"{indicators['volume_ratio']:.1f}x avg"
            },
            chart_patterns=[setup['name']],
            
            alerts_to_set=alerts,
            pre_entry_checklist=[
                "Confirm market is open and liquid",
                "Check for any pending news/earnings",
                "Verify option spread is reasonable (<10% of premium)",
                "Set all alerts before entry",
                "Have exit plan ready"
            ]
        )
    
    def _detect_macd_cross(self, macd_line: pd.Series, macd_signal: pd.Series) -> str:
        """Detect MACD crossover"""
        if len(macd_line) < 2:
            return 'none'
        
        if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-2] <= macd_signal.iloc[-2]:
            return 'bullish'
        elif macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_line.iloc[-2] >= macd_signal.iloc[-2]:
            return 'bearish'
        return 'none'
    
    def _calculate_trend_strength(self, close: pd.Series, sma20: pd.Series, sma50: pd.Series) -> float:
        """Calculate trend strength 0-100"""
        current = close.iloc[-1]
        
        if current > sma20.iloc[-1] > sma50.iloc[-1]:
            return 80
        elif current < sma20.iloc[-1] < sma50.iloc[-1]:
            return 20
        else:
            return 50
    
    def _select_strike(self, current_price: float, signal_type: str) -> float:
        """Select optimal strike price"""
        if signal_type == 'BUY_CALL':
            # Slightly OTM for better leverage
            strike = current_price * 1.01
        else:  # BUY_PUT
            strike = current_price * 0.99
        
        # Round to standard strikes
        if current_price < 50:
            return round(strike)
        elif current_price < 200:
            return round(strike / 5) * 5
        else:
            return round(strike / 10) * 10
    
    def _select_expiration(self, hold_days: int) -> str:
        """Select options expiration date"""
        target_date = datetime.now() + timedelta(days=hold_days + 3)
        
        # Find next Friday
        days_until_friday = (4 - target_date.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        
        expiration = target_date + timedelta(days=days_until_friday)
        return expiration.strftime('%Y-%m-%d')
    
    def _determine_entry_window(self, timing: str) -> Dict[str, str]:
        """Determine precise entry window"""
        now = datetime.now()
        
        if timing == 'immediate':
            return {
                'date': 'Today',
                'start_time': 'Now',
                'end_time': 'Next 30 minutes'
            }
        elif timing == 'morning_trend':
            return {
                'date': 'Next trading day',
                'start_time': '10:00 AM ET',
                'end_time': '11:30 AM ET'
            }
        elif timing == 'open_volatile':
            return {
                'date': 'Next trading day',
                'start_time': '9:45 AM ET',
                'end_time': '10:15 AM ET'
            }
        else:
            return {
                'date': 'Next trading day',
                'start_time': '10:00 AM ET',
                'end_time': '11:00 AM ET'
            }

def format_signal_alert(signal: PreciseOptionsSignal) -> str:
    """Format signal for display"""
    
    emoji = "🟢" if signal.signal_type == "BUY_CALL" else "🔴"
    
    alert = f"""
{emoji} {signal.symbol} - {signal.signal_type} - {signal.confidence}% Confidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 ENTRY:
   Trigger: ${signal.entry_trigger}
   Zone: ${signal.entry_zone[0]} - ${signal.entry_zone[1]}
   Time: {signal.entry_window['start_time']} - {signal.entry_window['end_time']}

📄 CONTRACT:
   Strike: ${signal.strike_price}
   Expiration: {signal.expiration_date}
   Max Premium: ${signal.max_premium}

🎯 TARGETS:
   Stop Loss: ${signal.stop_loss} (-{signal.stop_loss_pct}%)
   Target 1: ${signal.targets[0]['price']} (Exit {signal.targets[0]['exit_pct']}%)
   Target 2: ${signal.targets[1]['price']} (Exit {signal.targets[1]['exit_pct']}%)
   R:R Ratio: {signal.risk_reward_ratio}:1

⏰ TIMING:
   Hold: {signal.hold_duration}
   {signal.expiration_warning}

📊 SETUP: {signal.setup_name}
   RSI: {signal.key_indicators['RSI']}
   MACD: {signal.key_indicators['MACD']}
   Volume: {signal.key_indicators['Volume']}

🔔 SET THESE ALERTS NOW:
"""
    
    for alert in signal.alerts_to_set:
        alert += f"   • {alert}\n"
    
    return alert

# Example usage
if __name__ == "__main__":
    generator = PreciseSignalGenerator()
    
    # Analyze some symbols
    symbols = ['AAPL', 'TSLA', 'SPY', 'NVDA', 'AMD']
    
    print("🎯 Scanning for Precise Options Signals...")
    print("=" * 50)
    
    for symbol in symbols:
        signal = generator.analyze_symbol(symbol)
        if signal:
            print(format_signal_alert(signal))
            print("=" * 50) 