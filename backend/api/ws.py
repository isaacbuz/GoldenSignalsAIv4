from fastapi import APIRouter, WebSocket
from backend.agents.signal_engine import generate_ai_signal
from datetime import datetime
import asyncio
from typing import Set

router = APIRouter()
clients = set()
feedback_ws_clients: Set[WebSocket] = set()
event_ws_clients: Set[WebSocket] = set()

async def broadcast_feedback_event(event: dict):
    for ws in list(feedback_ws_clients):
        try:
            await ws.send_json(event)
        except Exception:
            feedback_ws_clients.discard(ws)

async def broadcast_agent_event(event: dict):
    for ws in list(event_ws_clients):
        try:
            await ws.send_json(event)
        except Exception:
            event_ws_clients.discard(ws)

@router.websocket("/ws/ai-signals")
async def ws_signal_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming normalized AI signals to frontend clients.
    Replace sample_data with real market data in production.
    """
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # Simulated market data (replace with real data feed)
            sample_data = {
                "price": [170 + i*0.2 for i in range(30)],
                "iv_history": [0.25, 0.3, 0.28, 0.35],
                "current_iv": 0.32,
            }
            signal_bundle = generate_ai_signal(sample_data, symbol="AAPL")
            await websocket.send_json(signal_bundle)
            await asyncio.sleep(15)
    except Exception as e:
        print("[WebSocket] Disconnected:", e)
    finally:
        clients.remove(websocket)

@router.websocket("/ws/feedback")
async def ws_feedback(websocket: WebSocket):
    await websocket.accept()
    feedback_ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(3600)  # keep alive
    except Exception as e:
        print("[WebSocket] Feedback disconnected:", e)
    finally:
        feedback_ws_clients.discard(websocket)

@router.websocket("/ws/agent-events")
async def ws_agent_events(websocket: WebSocket):
    await websocket.accept()
    event_ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(3600)
    except Exception as e:
        print("[WebSocket] Agent event disconnected:", e)
    finally:
        event_ws_clients.discard(websocket)
