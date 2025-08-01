"""
Frontend Logging API
Receives and stores frontend logs for debugging
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/logs", tags=["logging"])

# Log storage directory
LOG_DIR = Path("logs/frontend")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class LogEntry(BaseModel):
    timestamp: str
    level: str
    component: Optional[str]
    message: str
    data: Optional[dict]
    stack: Optional[str]
    userAgent: str
    url: str


class LogBatch(BaseModel):
    logs: List[LogEntry]
    sessionId: str
    timestamp: str


@router.post("/frontend")
async def receive_frontend_logs(batch: LogBatch):
    """
    Receive frontend logs and store them to files
    """
    try:
        # Create filename with date and session
        date_str = datetime.now().strftime("%Y%m%d")
        filename = LOG_DIR / f"frontend-{date_str}-{batch.sessionId}.log"

        # Append logs to file
        with open(filename, "a") as f:
            for log_entry in batch.logs:
                log_line = {
                    "timestamp": log_entry.timestamp,
                    "level": log_entry.level,
                    "component": log_entry.component,
                    "message": log_entry.message,
                    "data": log_entry.data,
                    "stack": log_entry.stack,
                    "userAgent": log_entry.userAgent,
                    "url": log_entry.url,
                    "sessionId": batch.sessionId,
                }
                f.write(json.dumps(log_line) + "\n")

        # Also write to a consolidated error log if there are errors
        errors = [log for log in batch.logs if log.level == "error"]
        if errors:
            error_filename = LOG_DIR / f"frontend-errors-{date_str}.log"
            with open(error_filename, "a") as f:
                for error in errors:
                    error_line = {
                        "timestamp": error.timestamp,
                        "component": error.component,
                        "message": error.message,
                        "data": error.data,
                        "stack": error.stack,
                        "url": error.url,
                        "sessionId": batch.sessionId,
                    }
                    f.write(json.dumps(error_line) + "\n")

        logger.info(f"Received {len(batch.logs)} frontend logs for session {batch.sessionId}")

        return {"status": "success", "count": len(batch.logs)}

    except Exception as e:
        logger.error(f"Error storing frontend logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store logs")


@router.get("/frontend/recent")
async def get_recent_logs(limit: int = 100):
    """
    Get recent frontend logs for debugging
    """
    try:
        # Find today's log files
        date_str = datetime.now().strftime("%Y%m%d")
        log_files = list(LOG_DIR.glob(f"frontend-{date_str}-*.log"))

        all_logs = []
        for log_file in log_files:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        all_logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue

        # Sort by timestamp and return most recent
        all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return all_logs[:limit]

    except Exception as e:
        logger.error(f"Error reading frontend logs: {str(e)}")
        return []


@router.delete("/frontend/clear")
async def clear_frontend_logs():
    """
    Clear all frontend logs (for testing)
    """
    try:
        for log_file in LOG_DIR.glob("frontend-*.log"):
            log_file.unlink()

        return {"status": "success", "message": "All frontend logs cleared"}

    except Exception as e:
        logger.error(f"Error clearing frontend logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear logs")
