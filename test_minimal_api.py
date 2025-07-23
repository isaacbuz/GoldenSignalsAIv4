#!/usr/bin/env python
"""Minimal FastAPI app for testing"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn
import json
import asyncio

app = FastAPI(title="GoldenSignalsAI Test API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GoldenSignalsAI Test API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "GoldenSignalsAI"}

@app.get("/api/v1/test")
async def test_endpoint():
    return {
        "message": "Test endpoint working",
        "frontend": "http://localhost:3000",

    }

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "action": "established",
            "timestamp": "2025-07-01T00:00:00Z",
            "data": {"message": "Connected to test WebSocket"}
        })

        # Keep connection alive and handle messages
        while True:
            # Wait for incoming message
            data = await websocket.receive_text()
            message = json.loads(data)

            # Echo back ping messages
            if message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": "2025-07-01T00:00:00Z"
                })

            # Handle subscription messages
            elif message.get("type") == "subscribe":
                await websocket.send_json({
                    "type": "subscription",
                    "action": "confirmed",
                    "topic": message.get("topic"),
                    "timestamp": "2025-07-01T00:00:00Z"
                })

            # Send mock signal updates periodically
            elif message.get("type") == "unsubscribe":
                await websocket.send_json({
                    "type": "subscription",
                    "action": "removed",
                    "topic": message.get("topic"),
                    "timestamp": "2025-07-01T00:00:00Z"
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    print("🚀 Starting minimal test API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Docs will be available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=8000)
