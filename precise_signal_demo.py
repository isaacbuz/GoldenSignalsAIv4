#!/usr/bin/env python3
"""
🎯 GoldenSignalsAI - Ultra-Precise Options Signals Demo
Shows exactly how specific our signals can be
"""

from datetime import datetime, timedelta
import json

def generate_precise_signal_examples():
    """Generate examples of ultra-precise options signals"""
    
    # Example 1: Bullish Signal
    call_signal = {
        "symbol": "AAPL",
        "signal_type": "BUY_CALL",
        "confidence": 78.5,
        "generated_at": "2024-01-10 09:15:00 ET",
        
        "timing": {
            "entry_date": "Today (Jan 10)",
            "entry_time": "10:00-10:30 AM ET",
            "entry_instruction": "Wait for first 30min candle to close above $186.50",
            "hold_duration": "2-3 days",
            "exit_by": "Friday Jan 12, 3:00 PM ET"
        },
        
        "options_contract": {
            "strike": 187.5,
            "expiration": "2024-01-19",
            "type": "Weekly",
            "recommended_price": "Max $2.50 per contract",
            "position_size": "10-20% of options capital"
        },
        
        "price_levels": {
            "current_price": 186.25,
            "entry_trigger": 186.50,
            "entry_range": [186.30, 186.70],
            "stop_loss": 184.80,
            "take_profit_1": 188.50,
            "take_profit_2": 190.25
        },
        
        "risk_management": {
            "stop_loss_pct": -0.8,
            "risk_per_trade": 500,
            "contracts_to_buy": 2,
            "max_loss": 500,
            "target_profit": 1000
        },
        
        "exit_rules": [
            "Exit 50% at $188.50 (Target 1)",
            "Exit remaining 50% at $190.25 (Target 2)",
            "Hard stop if AAPL drops below $184.80",
            "Time stop: Exit all by Friday 3:00 PM",
            "Exit if RSI hits 75+ (overbought)",
            "Exit on bearish MACD crossover"
        ],
        
        "technical_reasoning": {
            "setup": "Oversold bounce from support",
            "indicators": {
                "rsi": 28.5,
                "macd": "Bullish crossover yesterday",
                "support": 185.00,
                "resistance": 190.50,
                "volume": "1.5x average on bounce"
            },
            "pattern": "Double bottom at $185"
        },
        
        "alerts_to_set": [
            "Price alert at $186.50 (entry trigger)",
            "Price alert at $184.80 (stop loss)",
            "Price alert at $188.50 (first target)",
            "Time alert at 3:00 PM Friday"
        ]
    }
    
    # Example 2: Bearish Signal
    put_signal = {
        "symbol": "TSLA",
        "signal_type": "BUY_PUT",
        "confidence": 82.3,
        "generated_at": "2024-01-10 09:15:00 ET",
        
        "timing": {
            "entry_date": "Today (Jan 10)",
            "entry_time": "On breakdown below $242.50",
            "entry_instruction": "Enter on 15min candle close below support",
            "hold_duration": "1-2 days (quick move expected)",
            "exit_by": "Thursday Jan 11, EOD"
        },
        
        "options_contract": {
            "strike": 240.0,
            "expiration": "2024-01-12",
            "type": "Weekly (0DTE risk!)",
            "recommended_price": "Max $3.00 per contract",
            "position_size": "5-10% only (high risk)"
        },
        
        "price_levels": {
            "current_price": 243.20,
            "entry_trigger": 242.50,
            "entry_range": [242.30, 242.70],
            "stop_loss": 245.00,
            "take_profit_1": 239.50,
            "take_profit_2": 237.00
        },
        
        "risk_management": {
            "stop_loss_pct": -1.0,
            "risk_per_trade": 300,
            "contracts_to_buy": 1,
            "max_loss": 300,
            "target_profit": 600
        },
        
        "exit_rules": [
            "Exit 50% at $239.50",
            "Exit remaining at $237.00",
            "Stop loss if TSLA recovers above $245",
            "Time stop: Must exit by Thursday close",
            "Exit if RSI drops below 25 (oversold)",
            "Exit on bullish divergence"
        ],
        
        "technical_reasoning": {
            "setup": "Breakdown from rising wedge",
            "indicators": {
                "rsi": 72.8,
                "macd": "Bearish crossover today",
                "support": 237.00,
                "resistance": 245.00,
                "volume": "Heavy selling volume"
            },
            "pattern": "Failed breakout, bearish reversal"
        },
        
        "risk_warnings": [
            "⚠️ Weekly expiration - high theta decay",
            "⚠️ TSLA is volatile - use tight stops",
            "⚠️ Only risk what you can afford to lose"
        ]
    }
    
    # Example 3: High Conviction Signal
    high_conviction_signal = {
        "symbol": "SPY",
        "signal_type": "BUY_CALL",
        "confidence": 91.2,
        "generated_at": "2024-01-10 09:15:00 ET",
        
        "timing": {
            "entry_date": "Today - URGENT",
            "entry_time": "9:45-10:00 AM ET",
            "entry_instruction": "Buy on any dip to $451.50-452.00",
            "hold_duration": "3-5 days",
            "exit_by": "Next Tuesday latest"
        },
        
        "options_contract": {
            "strike": 453.0,
            "expiration": "2024-01-17",
            "type": "Weekly",
            "recommended_price": "Pay up to $3.50",
            "position_size": "20-30% (high conviction)"
        },
        
        "price_levels": {
            "current_price": 452.10,
            "entry_trigger": 451.80,
            "entry_range": [451.50, 452.20],
            "stop_loss": 449.50,
            "take_profit_1": 455.00,
            "take_profit_2": 457.50,
            "stretch_target": 460.00
        },
        
        "multi_scale_targets": {
            "scalp_exit": "30% at $453.50 (same day)",
            "swing_exit": "40% at $455.00 (1-2 days)",
            "runner": "30% at $457.50+ (3+ days)"
        },
        
        "advanced_exits": {
            "trailing_stop": "Activate at $455, trail by $1.50",
            "volatility_exit": "Exit if VIX spikes above 18",
            "time_decay_exit": "Reduce position by 50% on Thursday",
            "momentum_exit": "Exit if hourly RSI > 75"
        },
        
        "confluence_factors": [
            "✓ Bounce from 50-day MA",
            "✓ Bullish MACD cross on daily",
            "✓ VIX below 15 (low volatility)",
            "✓ Sector rotation into tech",
            "✓ Fed speaker dovish yesterday",
            "✓ Above all key moving averages"
        ]
    }
    
    return [call_signal, put_signal, high_conviction_signal]

def display_precise_signal(signal):
    """Display a precise signal in readable format"""
    
    # Determine emoji
    emoji = "🟢" if signal['signal_type'] == "BUY_CALL" else "🔴"
    
    print("\n" + "="*70)
    print(f"{emoji} {signal['symbol']} - {signal['signal_type']}")
    print(f"🎯 Confidence: {signal['confidence']:.1f}%")
    print("="*70)
    
    # EXACT TIMING
    print("\n⏰ PRECISE ENTRY TIMING:")
    timing = signal['timing']
    print(f"   📅 Date: {timing['entry_date']}")
    print(f"   🕐 Time: {timing['entry_time']}")
    print(f"   📋 Instructions: {timing['entry_instruction']}")
    print(f"   ⏳ Hold: {timing['hold_duration']}")
    print(f"   🚪 Exit by: {timing['exit_by']}")
    
    # SPECIFIC CONTRACT
    print("\n📄 EXACT OPTIONS CONTRACT:")
    contract = signal['options_contract']
    print(f"   Strike: ${contract['strike']}")
    print(f"   Expiration: {contract['expiration']} ({contract['type']})")
    print(f"   Max Price: {contract['recommended_price']}")
    print(f"   Position Size: {contract['position_size']}")
    
    # PRECISE LEVELS
    print("\n📍 EXACT PRICE LEVELS:")
    levels = signal['price_levels']
    print(f"   Current: ${levels['current_price']:.2f}")
    print(f"   Entry Trigger: ${levels['entry_trigger']:.2f}")
    print(f"   Entry Zone: ${levels['entry_range'][0]:.2f} - ${levels['entry_range'][1]:.2f}")
    print(f"   Stop Loss: ${levels['stop_loss']:.2f}")
    print(f"   Target 1: ${levels['take_profit_1']:.2f}")
    print(f"   Target 2: ${levels['take_profit_2']:.2f}")
    
    # RISK MANAGEMENT
    print("\n💰 POSITION SIZING:")
    risk = signal['risk_management']
    print(f"   Contracts: {risk['contracts_to_buy']}")
    print(f"   Max Risk: ${risk['max_loss']}")
    print(f"   Target Profit: ${risk['target_profit']}")
    print(f"   Risk/Reward: {risk['target_profit']/risk['max_loss']:.1f}:1")
    
    # EXIT RULES
    print("\n🚪 EXIT RULES (FOLLOW EXACTLY):")
    for i, rule in enumerate(signal['exit_rules'], 1):
        print(f"   {i}. {rule}")
    
    # REASONING
    print("\n📊 WHY THIS TRADE:")
    tech = signal['technical_reasoning']
    print(f"   Setup: {tech['setup']}")
    print(f"   Pattern: {tech['pattern']}")
    for key, value in tech['indicators'].items():
        print(f"   {key.upper()}: {value}")
    
    # ALERTS
    if 'alerts_to_set' in signal:
        print("\n🔔 ALERTS TO SET NOW:")
        for alert in signal['alerts_to_set']:
            print(f"   • {alert}")
    
    # WARNINGS
    if 'risk_warnings' in signal:
        print("\n⚠️ RISK WARNINGS:")
        for warning in signal['risk_warnings']:
            print(f"   {warning}")
    
    # CONFLUENCE (if high conviction)
    if 'confluence_factors' in signal:
        print("\n✨ CONFLUENCE FACTORS:")
        for factor in signal['confluence_factors']:
            print(f"   {factor}")

def main():
    """Demonstrate ultra-precise options signals"""
    
    print("🎯 GoldenSignalsAI - Ultra-Precise Options Signals")
    print("=" * 70)
    print("Showing exactly how specific our signals can be...\n")
    
    signals = generate_precise_signal_examples()
    
    for signal in signals:
        display_precise_signal(signal)
    
    # Show execution summary
    print("\n" + "="*70)
    print("📱 QUICK EXECUTION CHECKLIST")
    print("="*70)
    print("""
    1. ✓ Check current price vs entry trigger
    2. ✓ Set all price alerts immediately  
    3. ✓ Have broker ready with order template
    4. ✓ Confirm contract price is within range
    5. ✓ Set stop loss immediately after entry
    6. ✓ Follow exit rules exactly as specified
    7. ✓ Don't override the system!
    """)
    
    # Save to JSON
    with open('precise_signals_example.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    print("📄 Signals saved to precise_signals_example.json")
    print("="*70)

if __name__ == "__main__":
    main() 