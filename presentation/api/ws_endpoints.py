# ws_endpoints.py
# FastAPI WebSocket endpoint for real-time OHLCV and indicator data
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from application.services.data_service import DataService
from domain.trading.strategies.indicators import TechnicalIndicators
import asyncio
import json

router = APIRouter()
data_service = DataService()

clients = set()

# NOTE: Auth relaxed for testing. Add authentication logic here for production if needed.
@router.websocket("/ws/ohlcv")
async def websocket_ohlcv(websocket: WebSocket, symbol: str = 'AAPL', timeframe: str = 'D'):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # Fetch latest OHLCV data
            historical_df, _, _ = data_service.fetch_all_data(symbol)
            if historical_df is not None and len(historical_df) > 0:
                bars = [
                    {
                        "timestamp": str(row[0]),
                        "open": float(row[1]["open"]),
                        "high": float(row[1]["high"]),
                        "low": float(row[1]["low"]),
                        "close": float(row[1]["close"]),
                        "volume": float(row[1]["volume"])
                    }
                    for row in historical_df.reset_index().iterrows()
                ]
                # Compute indicators (last value only for demo)
                ti = TechnicalIndicators(historical_df)
                indicators = {}
                try:
                    indicators["MA_Confluence"] = float(ti.moving_average(20).iloc[-1])
                    indicators["RSI"] = float(ti.rsi().iloc[-1])
                    macd_line, _, _ = ti.macd(12, 26, 9)
                    indicators["MACD_Strength"] = float(macd_line.iloc[-1])
                    indicators["VWAP_Score"] = float(ti.vwap().iloc[-1]) if hasattr(ti, 'vwap') else None
                    indicators["Volume_Spike"] = float(historical_df['volume'].iloc[-1] / historical_df['volume'].rolling(20).mean().iloc[-1])
                except Exception as e:
                    indicators["error"] = str(e)
                payload = {
                    "bars": bars[-100:],  # send last 100 bars
                    "indicators": indicators
                }
                await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        clients.remove(websocket)
    except Exception:
        clients.remove(websocket)
        await websocket.close()
