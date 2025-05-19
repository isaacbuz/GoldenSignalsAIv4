#!/usr/bin/env python3
"""
Smart service launcher for GoldenSignalsAI
- Dynamically handles missing dependencies (Python & Node)
- Retries services on failure
- Logs errors and suggests code fixes
- Can be extended with AI/LLM for advanced error correction
"""
import subprocess
import sys
import os
import re
import time
from pathlib import Path

LOGFILE = "smart_run.log"
PYTHON_SERVICES = [
    {
        "name": "FastAPI backend",
        "cmd": ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        "cwd": str(Path.cwd()),
    },
    {
        "name": "API microservice",
        "cmd": ["uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "8080"],
        "cwd": str(Path.cwd()),
    },
]
NODE_SERVICES = [
    {
        "name": "Dash frontend",
        "cmd": ["npm", "start"],
        "cwd": str(Path("presentation/frontend").resolve()),
    },
]

MAX_RETRIES = 3
RETRY_WAIT = 3


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")


def install_python_package(package):
    log(f"Attempting to install missing Python package: {package}")
    res = subprocess.run([sys.executable, "-m", "pip", "install", package])
    return res.returncode == 0


def run_service(service):
    name = service["name"]
    cmd = service["cmd"]
    cwd = service.get("cwd", None)
    for attempt in range(1, MAX_RETRIES + 1):
        log(f"Starting {name} (attempt {attempt})...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=os.environ)
        out, err = proc.communicate()
        if proc.returncode == 0:
            log(f"{name} started successfully.")
            return True
        err_str = err.decode(errors="ignore")
        log(f"{name} failed to start. Error output:\n{err_str}")
        # Handle missing Python dependencies
        match = re.search(r"No module named '([^']+)'", err_str)
        if match:
            missing_pkg = match.group(1)
            if install_python_package(missing_pkg):
                log(f"Installed {missing_pkg}, retrying {name}...")
                continue
            else:
                log(f"Failed to install required package: {missing_pkg}")
                break
        # Handle Node.js errors (for npm)
        if "npm ERR!" in err_str or "command not found: npm" in err_str:
            log(f"Node.js/NPM error detected. Please ensure Node.js and npm are installed.")
            break
        # Handle Python syntax errors
        if "SyntaxError" in err_str:
            log(f"Syntax error detected in {name}. Please review the error above and fix the code.")
            break
        # Other fatal errors
        log(f"Retrying {name} in {RETRY_WAIT} seconds...")
        time.sleep(RETRY_WAIT)
    log(f"Giving up on {name} after {MAX_RETRIES} attempts.")
    return False


def main():
    log("=== Smart GoldenSignalsAI Service Launcher ===")
    # Ensure correct conda env
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name != "goldensignalsai":
        log(f"[WARNING] Not in 'goldensignalsai' conda environment. Current: {env_name}")
    # Run Python services
    for service in PYTHON_SERVICES:
        run_service(service)
    # Run Node/Dash services
    for service in NODE_SERVICES:
        run_service(service)
    log("All services attempted. Check above for errors or visit log file for details.")

if __name__ == "__main__":
    main()
