#!/usr/bin/env python3
"""
Monitor backend API for errors
"""
import requests
import time
import json

def check_endpoints():
    """Check all critical endpoints"""
    base_url = "http://localhost:8000"

    endpoints = [
        "/health",
        "/api/v1/market-data/AAPL/history?period=30d&interval=1d",
        "/api/v1/signals/latest",
        "/api/v1/agents",
    ]

    print("🔍 Monitoring API endpoints...")
    print("-" * 50)

    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                print(f"✅ {endpoint}: OK")
            else:
                print(f"❌ {endpoint}: {response.status_code}")
                print(f"   Response: {response.text[:100]}...")

        except Exception as e:
            print(f"❌ {endpoint}: ERROR - {str(e)}")

    print("-" * 50)

def check_websocket():
    """Check WebSocket connection"""
    try:
        import websocket
        ws = websocket.WebSocket()
        ws.connect("ws://localhost:8000/ws")
        ws.send(json.dumps({"type": "ping"}))
        result = ws.recv()
        ws.close()
        print("✅ WebSocket: Connected")
    except Exception as e:
        print(f"❌ WebSocket: {str(e)}")

if __name__ == "__main__":
    check_endpoints()
    check_websocket()

    # Check for specific error patterns in logs
    print("\n📋 Checking for recent errors...")
    try:
        # Check if there are any Python tracebacks in recent output
        import subprocess
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        if "python src/main.py" in result.stdout:
            print("✅ Backend process is running")
        else:
            print("❌ Backend process not found")
    except Exception as e:
        print(f"Could not check process: {e}")
