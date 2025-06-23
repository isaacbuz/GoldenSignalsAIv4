#!/usr/bin/env python3
"""
🎯 GoldenSignalsAI - Simple Signal Generator
Focused purely on generating entry/exit signals for options trading
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json

class SimpleSignalGenerator:
    """Streamlined signal generation - no complexity, just signals"""
    
    def __init__(self):
        self.signals = []
        
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Get market data"""
        try:
            data = yf.download(symbol, period="2mo", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data
        except:
            return pd.DataFrame()
    
    def calculate_signals(self, symbol: str) -> Dict:
        """Generate simple but effective signals"""
        data = self.fetch_data(symbol)
        
        if data.empty or len(data) < 50:
            return {
                "symbol": symbol,
                "signal": "NO_DATA",
                "confidence": 0
            }
        
        # Calculate indicators
        data['Returns'] = data['Close'].pct_change()
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['SMA_20']
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * bb_std
        data['BB_Lower'] = data['BB_Middle'] - 2 * bb_std
        
        # Volume
        data['Volume_Avg'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_Avg']
        
        # Get latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Signal logic
        signal_type = "HOLD"
        confidence = 0.5
        reasons = []
        
        # Bull signals
        bull_count = 0
        if latest['RSI'] < 30:
            bull_count += 1
            reasons.append("RSI oversold")
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            bull_count += 1
            reasons.append("Bullish MA alignment")
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            bull_count += 1
            reasons.append("MACD bullish crossover")
        if latest['Close'] < latest['BB_Lower']:
            bull_count += 1
            reasons.append("Below lower Bollinger Band")
        
        # Bear signals
        bear_count = 0
        if latest['RSI'] > 70:
            bear_count += 1
            reasons.append("RSI overbought")
        if latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            bear_count += 1
            reasons.append("Bearish MA alignment")
        if latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            bear_count += 1
            reasons.append("MACD bearish crossover")
        if latest['Close'] > latest['BB_Upper']:
            bear_count += 1
            reasons.append("Above upper Bollinger Band")
        
        # Volume confirmation
        if latest['Volume_Ratio'] > 1.5:
            reasons.append("High volume confirmation")
            confidence += 0.1
        
        # Determine signal
        if bull_count >= 2:
            signal_type = "BUY_CALL"
            confidence = min(0.5 + (bull_count * 0.15), 0.95)
        elif bear_count >= 2:
            signal_type = "BUY_PUT"
            confidence = min(0.5 + (bear_count * 0.15), 0.95)
        
        # Calculate levels
        atr = data['High'].rolling(14).mean() - data['Low'].rolling(14).mean()
        current_atr = atr.iloc[-1]
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal": signal_type,
            "confidence": confidence,
            "current_price": float(latest['Close']),
            "entry_zone": [
                float(latest['Close'] - current_atr * 0.3),
                float(latest['Close'] + current_atr * 0.3)
            ],
            "exit_target": float(latest['Close'] + current_atr * 1.5) if signal_type == "BUY_CALL" else float(latest['Close'] - current_atr * 1.5),
            "stop_loss": float(latest['Close'] - current_atr) if signal_type == "BUY_CALL" else float(latest['Close'] + current_atr),
            "indicators": {
                "rsi": float(latest['RSI']),
                "macd": float(latest['MACD']),
                "macd_signal": float(latest['MACD_Signal']),
                "sma_20": float(latest['SMA_20']),
                "sma_50": float(latest['SMA_50']),
                "volume_ratio": float(latest['Volume_Ratio'])
            },
            "reasons": reasons,
            "timeframe": "1-5 days"
        }

def main():
    """Generate signals for popular symbols"""
    generator = SimpleSignalGenerator()
    symbols = ['AAPL', 'SPY', 'TSLA', 'GOOGL', 'NVDA', 'AMZN', 'META', 'MSFT']
    
    print("="*60)
    print("🎯 GoldenSignalsAI - Options Trading Signals")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    all_signals = []
    
    for symbol in symbols:
        signal = generator.calculate_signals(symbol)
        all_signals.append(signal)
        
        if signal['signal'] == "NO_DATA":
            print(f"\n❌ {symbol}: No data available")
            continue
        
        # Display signal
        print(f"\n{'='*40}")
        print(f"📊 {symbol} - ${signal['current_price']:.2f}")
        print(f"{'='*40}")
        
        # Signal info
        emoji = "🟢" if signal['signal'] == "BUY_CALL" else "🔴" if signal['signal'] == "BUY_PUT" else "⚪"
        print(f"{emoji} Signal: {signal['signal']}")
        print(f"🎯 Confidence: {signal['confidence']:.1%}")
        
        if signal['signal'] != "HOLD":
            print(f"\n📍 Entry Zone: ${signal['entry_zone'][0]:.2f} - ${signal['entry_zone'][1]:.2f}")
            print(f"🎯 Target: ${signal['exit_target']:.2f}")
            print(f"🛑 Stop Loss: ${signal['stop_loss']:.2f}")
            
            # Risk/Reward calculation
            entry = signal['current_price']
            risk = abs(entry - signal['stop_loss'])
            reward = abs(signal['exit_target'] - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"📊 Risk/Reward: {rr_ratio:.2f}:1")
        
        # Indicators
        ind = signal['indicators']
        print(f"\n📈 Key Indicators:")
        print(f"   RSI: {ind['rsi']:.1f}")
        print(f"   MACD: {ind['macd']:.4f}")
        print(f"   Volume: {ind['volume_ratio']:.1f}x average")
        
        # Reasoning
        if signal['reasons']:
            print(f"\n💡 Signals: {', '.join(signal['reasons'])}")
    
    # Save signals
    with open('signals.json', 'w') as f:
        json.dump(all_signals, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Signals saved to signals.json")
    print("="*60)

if __name__ == "__main__":
    main() 