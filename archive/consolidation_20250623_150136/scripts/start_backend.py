#!/usr/bin/env python3
"""
Simple startup script for GoldenSignalsAI backend
"""
import subprocess
import sys
import os

def main():
    """Start the backend server."""
    print("🚀 Starting GoldenSignalsAI Backend...")
    print("=" * 50)
    
    # Ensure we're in the right directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python executable: {sys.executable}")
    print("=" * 50)
    
    # Start the backend using uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Backend stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 