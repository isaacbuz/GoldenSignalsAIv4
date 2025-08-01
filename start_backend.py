#!/usr/bin/env python
"""Temporary script to start backend with talib workaround"""

import os
import sys
import subprocess

print("🚀 Starting GoldenSignalsAI Backend...")

# Use the virtual environment's Python
venv_python = ".venv/bin/python"

# Check if venv exists
if not os.path.exists(venv_python):
    print("❌ Virtual environment not found. Please run: python -m venv .venv")
    sys.exit(1)

# Set environment to avoid numpy/talib conflicts
env = os.environ.copy()
env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

# Start uvicorn with the virtual environment
cmd = [
    venv_python,
    "-m",
    "uvicorn",
    "src.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload"
]

print(f"Running: {' '.join(cmd)}")
print("Backend will be available at: http://localhost:8000")
print("API docs will be available at: http://localhost:8000/docs")
print("\nPress Ctrl+C to stop the server")

try:
    subprocess.run(cmd, env=env)
except KeyboardInterrupt:
    print("\n✅ Backend stopped")
