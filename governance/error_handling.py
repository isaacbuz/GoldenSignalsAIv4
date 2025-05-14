import logging
import traceback
from typing import Dict, Any, Optional
from enum import Enum, auto

class ErrorSeverity(Enum):
    """Categorize error severity for appropriate handling."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class ApplicationError(Exception):
    """Base application-specific error with enhanced metadata."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an application-specific error.
        
        Args:
            message (str): Error description
            severity (ErrorSeverity): Error severity level
            context (Dict[str, Any], optional): Additional error context
        """
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.trace = traceback.format_exc()
        
        # Log the error
        logger = logging.getLogger('application_errors')
        log_method = {
            ErrorSeverity.LOW: logger.info,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }.get(severity, logger.error)
        
        log_method({
            'message': message,
            'severity': severity.name,
            'context': self.context,
            'trace': self.trace
        })

class ErrorHandler:
    """Centralized error management and recovery strategies."""
    
    @staticmethod
    def handle_error(
        error: Exception, 
        recovery_strategy: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Handle and potentially recover from an error.
        
        Args:
            error (Exception): The error to handle
            recovery_strategy (callable, optional): Function to attempt recovery
        
        Returns:
            Dict[str, Any]: Error report and recovery status
        """
        if isinstance(error, ApplicationError):
            severity = error.severity
        else:
            severity = ErrorSeverity.MEDIUM
        
        error_report = {
            'type': type(error).__name__,
            'message': str(error),
            'severity': severity.name,
            'recoverable': recovery_strategy is not None
        }
        
        try:
            if recovery_strategy:
                recovery_result = recovery_strategy(error)
                error_report['recovery_status'] = 'successful'
                error_report['recovery_details'] = recovery_result
            else:
                error_report['recovery_status'] = 'not_attempted'
        except Exception as recovery_error:
            error_report['recovery_status'] = 'failed'
            error_report['recovery_error'] = str(recovery_error)
        
        return error_report

# Example recovery strategies
def default_trading_recovery(error: Exception):
    """Default recovery for trading-related errors."""
    logging.warning(f"Trading error recovery initiated: {error}")
    return {
        'action': 'hold',
        'reason': 'Error in trading strategy'
    }

def data_fetch_recovery(error: Exception):
    """Recovery strategy for data fetching errors."""
    logging.warning(f"Data fetch error recovery: {error}")
    return {
        'fallback_source': 'local_cache',
        'retry_count': 3
    }
