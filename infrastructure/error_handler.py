import logging
import traceback
from typing import Callable, Any, Dict
from functools import wraps
import requests
from src.infrastructure.config_manager import config_manager
from src.infrastructure.monitoring import system_monitoring

class CircuitBreaker:
    """
    Advanced Circuit Breaker for external service calls
    
    Prevents cascading failures by:
    - Tracking service failures
    - Implementing exponential backoff
    - Providing fallback mechanisms
    """
    
    def __init__(
        self, 
        failure_threshold: int = 3, 
        recovery_time: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check circuit state
            if self.state == "OPEN":
                return self._fallback_handler(*args, **kwargs)
            
            try:
                result = func(*args, **kwargs)
                self._reset()
                return result
            
            except Exception as e:
                self._record_failure(e)
                raise
        
        return wrapper
    
    def _record_failure(self, exception: Exception):
        """
        Record and manage service failures
        
        Args:
            exception (Exception): Caught exception
        """
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        # Log failure details
        system_monitoring.capture_error(
            exception, 
            {"circuit_breaker_state": self.state}
        )
        
        # Check failure threshold
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
    
    def _reset(self):
        """Reset circuit breaker state"""
        self.failures = 0
        self.state = "CLOSED"
    
    def _fallback_handler(self, *args, **kwargs) -> Any:
        """
        Provide fallback mechanism when circuit is open
        
        Returns:
            Default or cached response
        """
        logging.warning("Circuit is OPEN. Returning fallback response.")
        return None

# Custom Exceptions
class ModelInferenceError(Exception):
    """Raised when a model inference operation fails."""
    pass

class DataFetchError(Exception):
    """Raised when data fetching fails."""
    pass

class ErrorHandler:
    """
    Comprehensive Error Handling and Recovery System
    
    Features:
    - Centralized error logging
    - Contextual error tracking
    - Automatic error reporting
    - Retry mechanisms
    """
    
    @staticmethod
    def handle_error(
        error: Exception, 
        context: Dict[str, Any] = None,
        retry: bool = False,
        max_retries: int = 3
    ) -> Any:
        """
        Centralized error handling method
        
        Args:
            error (Exception): Caught exception
            context (Dict[str, Any], optional): Error context
            retry (bool, optional): Enable retry mechanism
            max_retries (int, optional): Maximum retry attempts
        
        Returns:
            Retry result or None
        """
        # Granular error handling
        if isinstance(error, ModelInferenceError):
            logging.error(f"ModelInferenceError: {error}")
            system_monitoring.capture_error(error, context)
        elif isinstance(error, DataFetchError):
            logging.error(f"DataFetchError: {error}")
            system_monitoring.capture_error(error, context)
        else:
            logging.error(f"Unhandled error: {error}")
            system_monitoring.capture_error(error, context)
        
        # Optional retry mechanism
        if retry:
            return ErrorHandler._retry_operation(
                operation=context.get('operation'),
                max_retries=max_retries
            )
        
        return None

# Example usage:
# try:
#     ...
# except Exception as e:
#     ErrorHandler.handle_error(ModelInferenceError("Inference failed"), context={...})
    
    @staticmethod
    def _retry_operation(
        operation: Callable, 
        max_retries: int = 3,
        backoff_factor: float = 0.5
    ) -> Any:
        """
        Implement exponential backoff retry mechanism
        
        Args:
            operation (Callable): Operation to retry
            max_retries (int): Maximum retry attempts
            backoff_factor (float): Exponential backoff multiplier
        
        Returns:
            Operation result
        """
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
    
    @staticmethod
    def external_api_circuit_breaker(
        service_name: str
    ) -> CircuitBreaker:
        """
        Create circuit breaker for external API
        
        Args:
            service_name (str): Name of external service
        
        Returns:
            CircuitBreaker instance
        """
        return CircuitBreaker(
            failure_threshold=config_manager.get(
                f'error_handling.{service_name}.failure_threshold', 
                3
            ),
            recovery_time=config_manager.get(
                f'error_handling.{service_name}.recovery_time', 
                60
            )
        )

# Singleton instances
error_handler = ErrorHandler()
circuit_breaker = CircuitBreaker()
