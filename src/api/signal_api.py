"""
🎯 GoldenSignalsAI - Streamlined Signal API
Pure signal generation for options trading
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import sys
sys.path.append('..')
from demo_signal_system import DemoSignalGenerator

app = FastAPI(title="GoldenSignalsAI", version="3.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Signal model
class Signal(BaseModel):
    symbol: str
    timestamp: str
    signal: str  # BUY_CALL, BUY_PUT, HOLD
    confidence: float
    timeframe: str
    current_price: float
    entry_zone: List[float]
    exit_target: float
    stop_loss: float
    risk_reward_ratio: float
    indicators: Dict[str, float]
    reasoning: str
    signal_strength: str

# Initialize generator
signal_generator = DemoSignalGenerator()

@app.get("/")
def read_root():
    return {
        "name": "GoldenSignalsAI",
        "version": "3.0",
        "description": "Pure signal generation for options trading",
        "endpoints": {
            "/signals/{symbol}": "Get signal for specific symbol",
            "/signals": "Get signals for all symbols",
            "/health": "API health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/signals/{symbol}", response_model=Signal)
def get_signal(symbol: str):
    """Get trading signal for a specific symbol"""
    symbol = symbol.upper()
    
    try:
        signal = signal_generator.generate_demo_signal(symbol)
        return signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals", response_model=List[Signal])
def get_all_signals():
    """Get signals for all tracked symbols"""
    signals = []
    
    for symbol in signal_generator.symbols:
        try:
            signal = signal_generator.generate_demo_signal(symbol)
            signals.append(signal)
        except Exception as e:
            print(f"Error generating signal for {symbol}: {e}")
    
    return signals

@app.post("/signals/custom")
def get_custom_signals(symbols: List[str]):
    """Get signals for custom list of symbols"""
    signals = []
    
    for symbol in symbols:
        try:
            signal = signal_generator.generate_demo_signal(symbol.upper())
            signals.append(signal)
        except Exception as e:
            signals.append({
                "symbol": symbol,
                "error": str(e),
                "signal": "ERROR"
            })
    
    return signals

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GoldenSignalsAI Signal API...")
    print("📍 Access at: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 