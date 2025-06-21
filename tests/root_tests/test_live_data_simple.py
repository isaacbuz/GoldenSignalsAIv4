#!/usr/bin/env python3
"""
Simple Live Data Test - Yahoo Finance Only
Tests basic live data connection without complex dependencies
"""

import asyncio
import yfinance as yf
from datetime import datetime
import time

def get_live_quote(symbol):
    """Get live quote from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current data
        return {
            'symbol': symbol,
            'price': info.get('regularMarketPrice', 0),
            'bid': info.get('bid', 0),
            'ask': info.get('ask', 0),
            'volume': info.get('volume', 0),
            'previousClose': info.get('previousClose', 0),
            'dayHigh': info.get('dayHigh', 0),
            'dayLow': info.get('dayLow', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Error getting quote for {symbol}: {e}")
        return None

def get_options_data(symbol):
    """Get options chain from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            print(f"No options available for {symbol}")
            return None
            
        # Get first expiration
        exp = expirations[0]
        opt_chain = ticker.option_chain(exp)
        
        # Count calls and puts
        calls_count = len(opt_chain.calls)
        puts_count = len(opt_chain.puts)
        
        # Calculate put/call ratio
        call_volume = opt_chain.calls['volume'].sum()
        put_volume = opt_chain.puts['volume'].sum()
        pc_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        return {
            'expiration': exp,
            'calls_count': calls_count,
            'puts_count': puts_count,
            'call_volume': call_volume,
            'put_volume': put_volume,
            'put_call_ratio': pc_ratio
        }
    except Exception as e:
        print(f"Error getting options for {symbol}: {e}")
        return None

def calculate_simple_signal(quote_data):
    """Calculate a simple trading signal"""
    if not quote_data or quote_data['price'] == 0:
        return "HOLD", 0
        
    price = quote_data['price']
    prev_close = quote_data['previousClose']
    
    if prev_close == 0:
        return "HOLD", 0
        
    # Calculate percentage change
    change_pct = ((price - prev_close) / prev_close) * 100
    
    # Simple momentum signal
    if change_pct > 1.0:
        return "SELL", min(abs(change_pct) * 10, 80)  # Overbought
    elif change_pct < -1.0:
        return "BUY", min(abs(change_pct) * 10, 80)   # Oversold
    else:
        return "HOLD", 30

def main():
    """Main function to test live data"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║        GoldenSignalsAI - Simple Live Data Test             ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Testing Yahoo Finance connection (no API key required)    ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY']
    
    print(f"\n📊 Monitoring symbols: {', '.join(symbols)}")
    print("Press Ctrl+C to stop...\n")
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            for symbol in symbols:
                # Get quote
                quote = get_live_quote(symbol)
                
                if quote and quote['price'] > 0:
                    # Calculate signal
                    signal, confidence = calculate_simple_signal(quote)
                    
                    # Display results
                    print(f"\n📈 {symbol}")
                    print(f"   Price: ${quote['price']:.2f}")
                    print(f"   Bid/Ask: ${quote['bid']:.2f} / ${quote['ask']:.2f}")
                    print(f"   Volume: {quote['volume']:,}")
                    print(f"   Day Range: ${quote['dayLow']:.2f} - ${quote['dayHigh']:.2f}")
                    
                    # Signal
                    if signal == "BUY":
                        print(f"   🟢 Signal: {signal} (Confidence: {confidence}%)")
                    elif signal == "SELL":
                        print(f"   🔴 Signal: {signal} (Confidence: {confidence}%)")
                    else:
                        print(f"   ⚪ Signal: {signal} (Confidence: {confidence}%)")
                    
                    # Get options data
                    options = get_options_data(symbol)
                    if options:
                        print(f"   📊 Options: P/C Ratio = {options['put_call_ratio']:.2f}")
                        print(f"      Calls: {options['calls_count']} contracts, Volume: {options['call_volume']:,}")
                        print(f"      Puts: {options['puts_count']} contracts, Volume: {options['put_volume']:,}")
                else:
                    print(f"\n❌ {symbol}: No data available")
            
            # Wait before next update
            print(f"\n⏳ Next update in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✅ Test stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Test if yfinance is installed
    try:
        import yfinance
        print("✅ yfinance is installed")
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        exit(1)
        
    # Run the test
    main() 