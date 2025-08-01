#!/usr/bin/env python3
"""
Test API for frontend logging endpoint
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import json
import os
from pathlib import Path

app = FastAPI(title="Frontend Logging API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create logs directory
logs_dir = Path("logs/frontend")
logs_dir.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Frontend Logging API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/logs/frontend")
async def receive_frontend_logs(request: Request):
    """Receive and store frontend logs"""
    try:
        data = await request.json()

        # Add server timestamp
        log_entry = {
            **data,
            "server_timestamp": datetime.now().isoformat(),
            "client_ip": request.client.host
        }

        # Write to daily log file
        log_file = logs_dir / f"frontend-{datetime.now().strftime('%Y-%m-%d')}.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "status": "success",
            "message": "Log received",
            "timestamp": log_entry["server_timestamp"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }, 500

@app.get("/api/v1/test")
async def test_endpoint():
    return {"message": "Test endpoint working", "timestamp": datetime.now().isoformat()}

# WebSocket endpoint for signals
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Connected to signals WebSocket"
        }))

        while True:
            data = await websocket.receive_text()
            # Echo back or process
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Frontend Logging API...")
    print("📍 API endpoints:")
    print("   - POST http://localhost:8001/api/logs/frontend")
    print("   - WS   ws://localhost:8001/ws/signals")
    uvicorn.run(app, host="0.0.0.0", port=8001)
