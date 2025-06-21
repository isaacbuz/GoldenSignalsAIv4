#!/usr/bin/env python3
"""Test script to verify backend response format"""

import requests
import json

# Test the precise-options endpoint
response = requests.get("http://localhost:8000/api/v1/signals/precise-options?symbol=SPY&timeframe=15m")
data = response.json()

print("Response keys:", list(data.keys()))
print("\nNumber of signals:", len(data.get('signals', [])))

if data.get('signals'):
    first_signal = data['signals'][0]
    print("\nFirst signal keys:", list(first_signal.keys()))
    
    # Check for expected keys
    expected_keys = ['id', 'symbol', 'signal_type', 'type', 'confidence', 'entry_trigger', 'stop_loss', 'targets']
    missing_keys = [key for key in expected_keys if key not in first_signal]
    
    if missing_keys:
        print(f"\n❌ Missing expected keys: {missing_keys}")
        print("\n🔍 Actual keys found:", sorted(first_signal.keys()))
    else:
        print("\n✅ All expected keys found!")
        
    # Print first signal for inspection
    print("\nFirst signal (formatted):")
    print(json.dumps(first_signal, indent=2)) 