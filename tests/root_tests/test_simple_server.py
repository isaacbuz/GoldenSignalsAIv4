"""
Simple test server to verify FastAPI works
"""

import os
import sys

# Add src to path
sys.path.insert(0, 'src')

from fastapi import FastAPI
import uvicorn

# Create a simple app
app = FastAPI(title="Test Server")

@app.get("/")
def read_root():
    return {"message": "Simple test server is working!", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 Starting simple test server...")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
