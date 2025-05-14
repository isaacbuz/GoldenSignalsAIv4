# governance/audit.py
# Purpose: Implements audit logging for trading actions and system events to ensure
# transparency and compliance in options trading. Logs trade details, constraint violations,
# and system events to a persistent store.

import logging
import os
from datetime import datetime
from typing import Dict

import pandas as pd

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class AuditLogger:
    """Logs trading actions and system events for compliance and transparency."""

    def __init__(self, log_file: str = "audit_log.csv"):
        """Initialize with a log file for persistent storage.

        Args:
            log_file (str): Path to the audit log CSV file.
        """
        self.log_file = log_file
        self.columns = [
            "timestamp",
            "event_type",
            "symbol",
            "action",
            "details",
            "user_id",
        ]
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)
        logger.info({"message": f"AuditLogger initialized with log file: {log_file}"})

    def log_trade(self, trade: Dict, user_id: str, approved: bool):
        """Log a trade action (approved or rejected).

        Args:
            trade (Dict): Trade details with 'symbol', 'action', 'size', 'price'.
            user_id (str): User identifier.
            approved (bool): Whether the trade was approved.
        """
        logger.info({"message": f"Logging trade for user {user_id}: {trade}"})
        try:
            event_type = "TRADE_APPROVED" if approved else "TRADE_REJECTED"
            details = {
                "size": trade.get("size", 0),
                "price": trade.get("price", 0),
                "reason": "Approved" if approved else "Constraint violation",
            }
            self._write_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": event_type,
                    "symbol": trade.get("symbol", "UNKNOWN"),
                    "action": trade.get("action", "UNKNOWN"),
                    "details": str(details),
                    "user_id": user_id,
                }
            )
            logger.debug(
                {"message": f"Trade logged: {event_type} for {trade['symbol']}"}
            )
        except Exception as e:
            logger.error({"message": f"Failed to log trade: {str(e)}"})

    def log_system_event(self, event_type: str, details: Dict, user_id: str = "SYSTEM"):
        """Log a system event (e.g., agent failure, configuration change).

        Args:
            event_type (str): Type of event (e.g., 'AGENT_FAILURE').
            details (Dict): Event details.
            user_id (str): User identifier or 'SYSTEM' for system events.
        """
        logger.info({"message": f"Logging system event: {event_type}"})
        try:
            self._write_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": event_type,
                    "symbol": "N/A",
                    "action": "N/A",
                    "details": str(details),
                    "user_id": user_id,
                }
            )
            logger.debug({"message": f"System event logged: {event_type}"})
        except Exception as e:
            logger.error({"message": f"Failed to log system event: {str(e)}"})

    def _write_log(self, log_entry: Dict):
        """Write a log entry to the CSV file.

        Args:
            log_entry (Dict): Log entry with timestamp, event_type, symbol, action, details, user_id.
        """
        try:
            log_df = pd.DataFrame([log_entry], columns=self.columns)
            log_df.to_csv(self.log_file, mode="a", header=False, index=False)
        except Exception as e:
            logger.error({"message": f"Failed to write log entry: {str(e)}"})

    def get_audit_trail(
        self, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """Retrieve the audit trail for a specified date range.

        Args:
            start_date (str): Start date (ISO format, e.g., '2023-01-01').
            end_date (str): End date (ISO format).

        Returns:
            pd.DataFrame: Audit trail data.
        """
        logger.info(
            {"message": f"Retrieving audit trail from {start_date} to {end_date}"}
        )
        try:
            df = pd.read_csv(self.log_file)
            if start_date:
                df = df[df["timestamp"] >= start_date]
            if end_date:
                df = df[df["timestamp"] <= end_date]
            logger.debug({"message": f"Retrieved {len(df)} audit records"})
            return df
        except Exception as e:
            logger.error({"message": f"Failed to retrieve audit trail: {str(e)}"})
            return pd.DataFrame(columns=self.columns)
