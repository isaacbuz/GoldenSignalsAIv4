# tradingview_fetcher.py
# Fetches TradingView AI indicator signals via Node.js TradingView-API
import subprocess
import json
from datetime import datetime

def get_tradingview_signal(symbol="AAPL"):
    try:
        # Call the Node.js script to fetch the signal
        result = subprocess.run(
            ["node", "backend/agents/sources/tv_signal_fetch.js", symbol],
            capture_output=True, text=True, timeout=12
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"[TradingView] Error: {result.stderr}")
    except Exception as e:
        print(f"[TradingView] Exception: {e}")
    # Fallback to simulation if error occurs
    return {
        "symbol": symbol,
        "source": "TradingView_AI_Signals_V3",
        "signal": "buy",
        "confidence": 90,
        "type": "external",
        "indicator": "AI Signals V3",
        "rationale": "TV AI detected a bullish breakout + volume surge (simulated)",
        "timestamp": datetime.utcnow().isoformat()
    }
