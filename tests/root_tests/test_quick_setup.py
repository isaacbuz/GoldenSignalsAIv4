#!/usr/bin/env python3
"""
Quick Setup Test Script
Tests all components are working
"""

import requests
import json
from datetime import datetime

print("🔍 GoldenSignalsAI Quick Setup Test\n" + "="*50)

# 1. Test Backend API
print("\n1️⃣ Testing Backend API...")
try:
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ Backend is running!")
        print(f"   Response: {response.json()}")
    else:
        print("❌ Backend returned error:", response.status_code)
except Exception as e:
    print("❌ Backend not accessible:", str(e))

# 2. Test Market Data Endpoint
print("\n2️⃣ Testing Market Data API...")
try:
    response = requests.get("http://localhost:8000/api/v1/market-data/AAPL")
    if response.status_code == 200:
        data = response.json()
        print("✅ Market data endpoint working!")
        print(f"   AAPL Price: ${data.get('price', 'N/A')}")
    else:
        print("❌ Market data error:", response.status_code)
except Exception as e:
    print("❌ Market data not accessible:", str(e))

# 3. Test Signals Endpoint
print("\n3️⃣ Testing Signals API...")
try:
    response = requests.get("http://localhost:8000/api/v1/signals/AAPL")
    if response.status_code == 200:
        signal = response.json()
        print("✅ Signals endpoint working!")
        print(f"   Signal: {signal.get('signal', 'N/A')}")
        print(f"   Confidence: {signal.get('confidence', 0)*100:.1f}%")
    else:
        print("❌ Signals error:", response.status_code)
except Exception as e:
    print("❌ Signals not accessible:", str(e))

# 4. Test Latest Signals
print("\n4️⃣ Testing Latest Signals API...")
try:
    response = requests.get("http://localhost:8000/api/v1/signals/latest?limit=5")
    if response.status_code == 200:
        signals = response.json()
        print(f"✅ Found {len(signals)} latest signals!")
        for sig in signals[:3]:
            print(f"   - {sig.get('symbol', 'N/A')}: {sig.get('signal_type', 'N/A')} ({sig.get('confidence', 0)*100:.0f}%)")
    else:
        print("❌ Latest signals error:", response.status_code)
except Exception as e:
    print("❌ Latest signals not accessible:", str(e))

# 5. Test yfinance directly
print("\n5️⃣ Testing YFinance...")
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1d")
    if not hist.empty:
        print("✅ YFinance working!")
        print(f"   AAPL Close: ${hist['Close'].iloc[-1]:.2f}")
    else:
        print("⚠️ YFinance returned no data (market might be closed)")
except Exception as e:
    print("❌ YFinance error:", str(e))

# 6. Check Frontend
print("\n6️⃣ Frontend Status...")
print("   Check: http://localhost:5173")
print("   If not working, run: cd frontend && npm run dev")

# Summary
print("\n" + "="*50)
print("📋 Setup Summary:")
print("   - Backend API: Check above results")
print("   - Frontend: Visit http://localhost:5173")
print("   - API Docs: Visit http://localhost:8000/docs")
print("\n💡 Next Steps:")
print("   1. If backend not running: cd src && python main_simple.py")
print("   2. If frontend not running: cd frontend && npm run dev")
print("   3. Check MVP_IMPLEMENTATION_GUIDE.md for detailed instructions") 