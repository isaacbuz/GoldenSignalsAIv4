"""
Sentry Error Tracking Integration
Provides comprehensive error tracking and monitoring for the application
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional

import sentry_sdk
from sentry_sdk import capture_exception, capture_message, set_context, set_tag, set_user
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

logger = logging.getLogger(__name__)


class ErrorTrackingService:
    """Centralized error tracking service using Sentry"""

    def __init__(self, dsn: Optional[str] = None):
        """
        Initialize Sentry error tracking

        Args:
            dsn: Sentry DSN (Data Source Name)
        """
        self.dsn = dsn or os.getenv("SENTRY_DSN")
        self.environment = os.getenv("API_ENV", "development")
        self.enabled = bool(self.dsn) and self.environment != "development"

        if self.enabled:
            self._initialize_sentry()
        else:
            logger.info("⚠️ Sentry error tracking disabled (no DSN or in development)")

    def _initialize_sentry(self) -> None:
        """Initialize Sentry with custom configuration"""
        try:
            # Configure logging integration
            logging_integration = LoggingIntegration(
                level=logging.INFO,  # Capture info and above as breadcrumbs
                event_level=logging.ERROR,  # Send errors as events
            )

            sentry_sdk.init(
                dsn=self.dsn,
                environment=self.environment,
                integrations=[
                    FastApiIntegration(transaction_style="endpoint"),
                    SqlalchemyIntegration(),
                    RedisIntegration(),
                    logging_integration,
                ],
                # Performance monitoring
                traces_sample_rate=0.1 if self.environment == "production" else 1.0,
                # Session tracking
                release=os.getenv("APP_VERSION", "unknown"),
                # Additional options
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send personally identifiable information
                before_send=self._before_send,
                # Breadcrumbs
                max_breadcrumbs=50,
                # Request bodies
                request_bodies="medium",
                # Sampling
                sample_rate=1.0 if self.environment != "production" else 0.5,
            )

            # Set global tags
            set_tag("service", "goldensignals-api")
            set_tag("component", "backend")

            logger.info(f"✅ Sentry initialized for {self.environment} environment")

        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
            self.enabled = False

    def _before_send(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process events before sending to Sentry

        Args:
            event: The event dictionary
            hint: Additional information about the event

        Returns:
            Modified event or None to drop it
        """
        # Filter out certain errors in development
        if self.environment == "development":
            error_type = hint.get("exc_info", [None])[0]
            if error_type and error_type.__name__ in ["KeyboardInterrupt", "SystemExit"]:
                return None

        # Add custom context
        if "request" in event:
            # Sanitize sensitive data from request
            if "headers" in event["request"]:
                event["request"]["headers"] = self._sanitize_headers(event["request"]["headers"])
            if "data" in event["request"]:
                event["request"]["data"] = self._sanitize_data(event["request"]["data"])

        return event

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive headers"""
        sensitive_headers = ["authorization", "cookie", "x-api-key", "x-auth-token"]
        return {
            k: v if k.lower() not in sensitive_headers else "[REDACTED]" for k, v in headers.items()
        }

    def _sanitize_data(self, data: Any) -> Any:
        """Remove sensitive data fields"""
        if isinstance(data, dict):
            sensitive_fields = ["password", "token", "secret", "api_key", "credit_card"]
            return {
                k: "[REDACTED]" if any(field in k.lower() for field in sensitive_fields) else v
                for k, v in data.items()
            }
        return data

    def capture_exception(self, exception: Exception, **kwargs) -> Optional[str]:
        """
        Capture an exception and send to Sentry

        Args:
            exception: The exception to capture
            **kwargs: Additional context

        Returns:
            Event ID if sent, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Add custom context
            for key, value in kwargs.items():
                set_context(key, value)

            event_id = capture_exception(exception)
            logger.error(f"Exception captured: {event_id} - {exception}")
            return event_id

        except Exception as e:
            logger.error(f"Failed to capture exception: {e}")
            return None

    def capture_message(self, message: str, level: str = "info", **kwargs) -> Optional[str]:
        """
        Capture a message and send to Sentry

        Args:
            message: The message to capture
            level: Severity level
            **kwargs: Additional context

        Returns:
            Event ID if sent, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Add custom context
            for key, value in kwargs.items():
                set_context(key, value)

            event_id = capture_message(message, level=level)
            return event_id

        except Exception as e:
            logger.error(f"Failed to capture message: {e}")
            return None

    def set_user_context(
        self, user_id: str, email: Optional[str] = None, username: Optional[str] = None, **kwargs
    ) -> None:
        """Set user context for error tracking"""
        if not self.enabled:
            return

        set_user({"id": user_id, "email": email, "username": username, **kwargs})

    def set_trading_context(self, symbol: str, action: str, confidence: float, **kwargs) -> None:
        """Set trading-specific context"""
        if not self.enabled:
            return

        set_context(
            "trading", {"symbol": symbol, "action": action, "confidence": confidence, **kwargs}
        )

    def set_agent_context(
        self, agent_name: str, agent_type: str, performance: Dict[str, Any]
    ) -> None:
        """Set agent-specific context"""
        if not self.enabled:
            return

        set_context("agent", {"name": agent_name, "type": agent_type, "performance": performance})


# Global error tracking instance
_error_tracker: Optional[ErrorTrackingService] = None


def get_error_tracker() -> ErrorTrackingService:
    """Get or create error tracking service singleton"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTrackingService()
    return _error_tracker


# Decorator for automatic error tracking
def track_errors(operation: str = "operation"):
    """
    Decorator to automatically track errors in functions

    Args:
        operation: Name of the operation being performed
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_error_tracker()

            try:
                # Set operation context
                set_context(
                    "operation",
                    {"name": operation, "function": func.__name__, "module": func.__module__},
                )

                # Execute function
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                # Capture exception with context
                tracker.capture_exception(
                    e,
                    operation=operation,
                    function=func.__name__,
                    args=str(args)[:200],  # Limit size
                    kwargs=str(kwargs)[:200],
                )
                # Re-raise the exception
                raise

        return wrapper

    return decorator


# FastAPI exception handler
def create_sentry_exception_handler(tracker: ErrorTrackingService):
    """Create a FastAPI exception handler that reports to Sentry"""

    async def sentry_exception_handler(request, exc):
        """Handle exceptions and report to Sentry"""
        # Capture the exception
        tracker.capture_exception(
            exc,
            request={
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
            },
        )

        # Return appropriate response
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Our team has been notified.",
                "type": type(exc).__name__,
            },
        )

    return sentry_exception_handler
