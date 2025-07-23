"""
🚀 Integrated Signals API
FastAPI endpoints for precise options and arbitrage signals
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agents.signals.arbitrage_signals import ArbitrageSignal
from agents.signals.integrated_signal_system import IntegratedSignalSystem
from agents.signals.precise_options_signals import PreciseOptionsSignal

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])

# Global signal system instance
signal_system = IntegratedSignalSystem()

# Request/Response Models
class SignalScanRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to scan")
    include_options: bool = Field(True, description="Include options signals")
    include_arbitrage: bool = Field(True, description="Include arbitrage signals")
    min_confidence: float = Field(70.0, description="Minimum confidence level")

class OptionsSignalResponse(BaseModel):
    signal_id: str
    symbol: str
    signal_type: str
    confidence: float
    strike_price: float
    expiration_date: str
    entry_trigger: float
    stop_loss: float
    targets: List[Dict[str, float]]
    risk_reward_ratio: float
    entry_window: Dict[str, str]

class ArbitrageSignalResponse(BaseModel):
    signal_id: str
    arb_type: str
    primary_asset: str
    confidence: float
    spread_pct: float
    estimated_profit: float
    risk_level: str
    holding_period: str
    execution_steps: List[str]

class SignalScanResponse(BaseModel):
    scan_timestamp: str
    total_signals: int
    options_signals: List[OptionsSignalResponse]
    arbitrage_signals: List[ArbitrageSignalResponse]
    combined_signals: List[Dict]

class ExecutionPlanRequest(BaseModel):
    risk_tolerance: str = Field("MEDIUM", description="LOW, MEDIUM, or HIGH")
    capital: float = Field(10000, description="Available capital")
    signal_types: Optional[List[str]] = Field(None, description="Filter by signal types")
    max_positions: int = Field(5, description="Maximum concurrent positions")

@router.post("/scan", response_model=SignalScanResponse)
async def scan_markets(request: SignalScanRequest):
    """
    Scan markets for precise options and arbitrage signals
    """
    try:
        # Configure what to scan
        types_to_scan = []
        if request.include_options:
            types_to_scan.append('options')
        if request.include_arbitrage:
            types_to_scan.append('arbitrage')
        
        # Run comprehensive scan
        signals = await signal_system.scan_all_markets(request.symbols)
        
        # Filter by confidence
        options_signals = [
            s for s in signals.get('options', [])
            if s.confidence >= request.min_confidence
        ]
        
        arbitrage_signals = [
            s for s in signals.get('arbitrage', [])
            if s.confidence >= request.min_confidence
        ]
        
        # Convert to response format
        options_responses = [
            OptionsSignalResponse(
                signal_id=s.signal_id,
                symbol=s.symbol,
                signal_type=s.signal_type,
                confidence=s.confidence,
                strike_price=s.strike_price,
                expiration_date=s.expiration_date,
                entry_trigger=s.entry_trigger,
                stop_loss=s.stop_loss,
                targets=s.targets,
                risk_reward_ratio=s.risk_reward_ratio,
                entry_window=s.entry_window
            ) for s in options_signals[:10]  # Limit to 10
        ]
        
        arbitrage_responses = [
            ArbitrageSignalResponse(
                signal_id=s.signal_id,
                arb_type=s.arb_type,
                primary_asset=s.primary_asset,
                confidence=s.confidence,
                spread_pct=s.spread_pct,
                estimated_profit=s.estimated_profit,
                risk_level=s.risk_level,
                holding_period=s.holding_period,
                execution_steps=s.execution_steps[:3]  # Top 3 steps
            ) for s in arbitrage_signals[:10]  # Limit to 10
        ]
        
        return SignalScanResponse(
            scan_timestamp=datetime.now().isoformat(),
            total_signals=len(options_responses) + len(arbitrage_responses),
            options_signals=options_responses,
            arbitrage_signals=arbitrage_responses,
            combined_signals=signals.get('combined', [])[:5]  # Top 5
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime/{symbol}")
async def get_realtime_signal(symbol: str):
    """
    Get real-time signal for a specific symbol
    """
    try:
        # Quick scan for single symbol
        signals = await signal_system.scan_all_markets([symbol])
        
        response = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "options_signal": None,
            "arbitrage_opportunities": [],
            "combined_strategy": None
        }
        
        # Get first options signal
        if signals.get('options'):
            signal = signals['options'][0]
            response["options_signal"] = {
                "type": signal.signal_type,
                "confidence": signal.confidence,
                "entry": signal.entry_trigger,
                "stop": signal.stop_loss,
                "targets": signal.targets,
                "strike": signal.strike_price,
                "expiration": signal.expiration_date
            }
        
        # Get arbitrage opportunities
        for arb in signals.get('arbitrage', [])[:3]:
            if arb.primary_asset == symbol:
                response["arbitrage_opportunities"].append({
                    "type": arb.arb_type,
                    "spread": f"{arb.spread_pct:.2f}%",
                    "profit": arb.estimated_profit,
                    "risk": arb.risk_level
                })
        
        # Get combined strategy
        for combined in signals.get('combined', []):
            if combined.get('symbol') == symbol:
                response["combined_strategy"] = combined
                break
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execution-plan")
async def generate_execution_plan(request: ExecutionPlanRequest):
    """
    Generate detailed execution plan based on user profile
    """
    try:
        # Get top opportunities
        opportunities = signal_system.get_top_opportunities(
            risk_tolerance=request.risk_tolerance,
            capital=request.capital,
            types=request.signal_types
        )
        
        # Limit to max positions
        opportunities = opportunities[:request.max_positions]
        
        # Generate execution plan
        plan = signal_system.generate_execution_plan(opportunities)
        
        # Add user preferences
        plan["user_profile"] = {
            "risk_tolerance": request.risk_tolerance,
            "capital": request.capital,
            "max_positions": request.max_positions
        }
        
        return plan
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active")
async def get_active_signals():
    """
    Get all currently active signals
    """
    try:
        active = signal_system.active_signals
        
        return {
            "timestamp": datetime.now().isoformat(),
            "counts": {
                "options": len(active.get('options', [])),
                "arbitrage": len(active.get('arbitrage', [])),
                "combined": len(active.get('combined', []))
            },
            "signals": {
                "options": [
                    {
                        "symbol": s.symbol,
                        "type": s.signal_type,
                        "confidence": s.confidence,
                        "entry": s.entry_trigger,
                        "targets": s.targets
                    } for s in active.get('options', [])[:5]
                ],
                "arbitrage": [
                    {
                        "asset": s.primary_asset,
                        "type": s.arb_type,
                        "spread": f"{s.spread_pct:.2f}%",
                        "profit": s.estimated_profit
                    } for s in active.get('arbitrage', [])[:5]
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/paper-trade")
async def execute_paper_trade(plan: Dict):
    """
    Execute trades in paper trading mode
    """
    try:
        results = await signal_system.execute_paper_trades(plan)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/backtest")
async def backtest_signals(
    symbols: List[str] = Query(...),
    days: int = Query(30, description="Days to backtest"),
    strategy: str = Query("ALL", description="Strategy type to test")
):
    """
    Backtest signal performance
    """
    try:
        # This would connect to backtesting engine
        # For demo, return mock results
        return {
            "period": f"Last {days} days",
            "symbols_tested": symbols,
            "strategy": strategy,
            "results": {
                "total_signals": 127,
                "win_rate": 68.5,
                "avg_return": 3.2,
                "sharpe_ratio": 1.8,
                "max_drawdown": -8.5,
                "best_trade": {
                    "symbol": "NVDA",
                    "return": 15.3,
                    "type": "BUY_CALL"
                },
                "worst_trade": {
                    "symbol": "TSLA", 
                    "return": -6.2,
                    "type": "BUY_PUT"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time signal updates
    """
    await websocket.accept()
    
    try:
        while True:
            # Send updates every 5 seconds
            await asyncio.sleep(5)
            
            # Get latest signals
            active = signal_system.active_signals
            
            update = {
                "type": "signal_update",
                "timestamp": datetime.now().isoformat(),
                "new_signals": {
                    "options": len(active.get('options', [])),
                    "arbitrage": len(active.get('arbitrage', []))
                }
            }
            
            await websocket.send_json(update)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# Background task to continuously scan markets
async def continuous_market_scan():
    """
    Background task to continuously scan markets
    """
    popular_symbols = [
        'TSLA', 'AAPL', 'NVDA', 'AMD', 'SPY', 'QQQ',
        'MSFT', 'GOOGL', 'META', 'AMZN'
    ]
    
    while True:
        try:
            # Scan every minute
            await signal_system.scan_all_markets(popular_symbols)
            await asyncio.sleep(60)
        except Exception as e:
            print(f"Background scan error: {e}")
            await asyncio.sleep(60)

# Start background scanning on startup
@router.on_event("startup")
async def startup_event():
    """
    Start background tasks on API startup
    """
    asyncio.create_task(continuous_market_scan()) 