#!/usr/bin/env python3
"""
Test live data fetching with yfinance
"""

import yfinance as yf
import asyncio
from datetime import datetime

async def test_live_data():
    """Test fetching live data for popular stocks"""
    
    symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
    
    print("=" * 60)
    print("🚀 Testing Live Market Data with yfinance")
    print("=" * 60)
    print()
    
    for symbol in symbols:
        try:
            # Get ticker
            ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Extract key data
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            previous_close = info.get("previousClose", 0)
            volume = info.get("volume") or info.get("regularMarketVolume", 0)
            market_cap = info.get("marketCap", 0)
            
            # Calculate change
            change = current_price - previous_close if current_price and previous_close else 0
            change_percent = (change / previous_close * 100) if previous_close else 0
            
            print(f"📊 {symbol}:")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Change: ${change:.2f} ({change_percent:+.2f}%)")
            print(f"   Volume: {volume:,}")
            print(f"   Market Cap: ${market_cap:,.0f}")
            print()
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
            print()
    
    # Test historical data
    print("-" * 60)
    print("📈 Testing Historical Data (AAPL - Last 5 days)")
    print("-" * 60)
    
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d", interval="1d")
        
        if not hist.empty:
            print("\nDate         Open     High     Low      Close    Volume")
            print("-" * 60)
            for index, row in hist.iterrows():
                print(f"{index.strftime('%Y-%m-%d')}   "
                      f"{row['Open']:7.2f}  "
                      f"{row['High']:7.2f}  "
                      f"{row['Low']:7.2f}  "
                      f"{row['Close']:7.2f}  "
                      f"{int(row['Volume']):,}")
        else:
            print("No historical data available")
            
    except Exception as e:
        print(f"❌ Error fetching historical data: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Live data test complete!")
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_live_data()) 