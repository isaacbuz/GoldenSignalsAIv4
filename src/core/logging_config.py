"""
Logging Configuration for GoldenSignalsAI V3

Centralized logging setup with structured logging, multiple handlers,
and performance monitoring.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import settings


def setup_logging() -> logger:
    """
    Configure logging for the application

    Returns:
        loguru.logger: Configured logger instance
    """
    # Remove default loguru handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console handler with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=settings.monitoring.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler for general logs
    logger.add(
        log_dir / "goldensignals.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        serialize=False,
    )

    # File handler for errors only
    logger.add(
        log_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="gz",
        serialize=True,  # JSON format for error analysis
        backtrace=True,
        diagnose=True,
    )

    # File handler for agent performance logs
    logger.add(
        log_dir / "agent_performance.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="50 MB",
        retention="60 days",
        filter=lambda record: "agent_performance" in record.get("extra", {}),
        serialize=True,
    )

    # File handler for trading signals
    logger.add(
        log_dir / "signals.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="200 MB",
        retention="180 days",
        filter=lambda record: "signal" in record.get("extra", {}),
        serialize=True,
    )

    # Audit log for critical operations
    logger.add(
        log_dir / "audit.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="365 days",
        filter=lambda record: "audit" in record.get("extra", {}),
        serialize=True,
    )

    # Configure standard Python logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Replace standard logging handlers
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    logger.info("Logging configuration initialized")
    return logger


def log_signal(signal_data: dict, agent_name: str) -> None:
    """
    Log trading signal with structured data

    Args:
        signal_data: Signal information
        agent_name: Name of the agent that generated the signal
    """
    logger.bind(signal=True).info(
        "Signal generated",
        extra={
            "signal": True,
            "agent": agent_name,
            "symbol": signal_data.get("symbol"),
            "signal_type": signal_data.get("signal_type"),
            "confidence": signal_data.get("confidence"),
            "timestamp": signal_data.get("created_at"),
        },
    )


def log_agent_performance(agent_name: str, performance_data: dict) -> None:
    """
    Log agent performance metrics

    Args:
        agent_name: Name of the agent
        performance_data: Performance metrics
    """
    logger.bind(agent_performance=True).info(
        "Agent performance update",
        extra={
            "agent_performance": True,
            "agent": agent_name,
            "accuracy": performance_data.get("accuracy"),
            "total_signals": performance_data.get("total_signals"),
            "avg_confidence": performance_data.get("avg_confidence"),
        },
    )


def log_audit(action: str, user_id: Optional[str], details: dict) -> None:
    """
    Log audit trail for critical operations

    Args:
        action: Action performed
        user_id: User who performed the action
        details: Additional details
    """
    logger.bind(audit=True).info(
        f"Audit: {action}",
        extra={"audit": True, "action": action, "user_id": user_id, "details": details},
    )


def log_error_with_context(error: Exception, context: dict) -> None:
    """
    Log error with additional context

    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger.bind(**context).error(f"Error occurred: {str(error)}", extra=context)
