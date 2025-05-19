# infrastructure/error_handling.py

import logging
import traceback

class ApplicationError(Exception):
    """Generic application-level error for GoldenSignalsAI."""
    def __init__(self, message, severity=None, context=None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.trace = traceback.format_exc()
        # Optionally log the error
        logger = logging.getLogger('application_errors')
        logger.error({
            'message': message,
            'severity': getattr(severity, 'name', str(severity)),
            'context': self.context,
            'trace': self.trace
        })

class ErrorSeverity:
    """Severity levels for application errors."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'

class ErrorHandler:
    @staticmethod
    def handle(error):
        print(f"Error: {error}")
