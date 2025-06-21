#!/usr/bin/env python3
"""
Quick fix for yfinance multi-column issue
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test the issue
symbol = 'AAPL'
end = datetime.now()
start = end - timedelta(days=30)

print("Testing yfinance download...")
data = yf.download(symbol, start=start, end=end, progress=False)
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns}")
print(f"Is MultiIndex: {isinstance(data.columns, pd.MultiIndex)}")

if isinstance(data.columns, pd.MultiIndex):
    print("\nFixing MultiIndex columns...")
    # For single symbol, just drop the ticker level
    data.columns = data.columns.droplevel(1)
    print(f"Fixed columns: {data.columns}")

# Test indicator calculation
print("\nTesting Bollinger Bands calculation...")
df = data.copy()
df['BB_Middle'] = df['Close'].rolling(20).mean()
bb_std = df['Close'].rolling(20).std()
print(f"bb_std type: {type(bb_std)}")
print(f"BB_Middle type: {type(df['BB_Middle'])}")

# This should work
df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std

print("✅ Bollinger Bands calculated successfully!") 