import asyncio
import logging
from typing import Any, Dict, List
from fastapi import FastAPI, Depends

from dependency_injector.wiring import inject, Provide
from infrastructure.dependency_container import Container
from infrastructure.config_manager import config_manager
from infrastructure.error_handling import ErrorHandler
from optimization.performance_monitor import performance_monitor

# Create FastAPI app
app = FastAPI(
    title="GoldenSignalsAI", 
    description="AI-powered Options Trading Signal Generator"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@app.get("/", tags=["Root"])
@performance_monitor.track_performance
def read_root():
    """
    Root endpoint providing system information.
    
    Returns:
        Dict[str, str]: System welcome message and configuration details
    """
    try:
        return {
            "message": "Welcome to GoldenSignalsAI",
            "version": config_manager.get('version', '1.0.0'),
            "environment": config_manager.get('environment', 'development')
        }
    except Exception as e:
        error_report = ErrorHandler.handle_error(e)
        return {"error": error_report}

@app.get("/health", tags=["Monitoring"])
@performance_monitor.track_performance
@inject
def health_check(
    signal_engine=Depends(Provide[Container.signal_engine])
):
    """
    Health check endpoint for system diagnostics.
    
    Returns:
        Dict[str, Any]: System health status and performance metrics
    """
    try:
        performance_summary = performance_monitor.get_performance_summary()
        return {
            "status": "healthy",
            "performance": performance_summary
        }
    except Exception as e:
        error_report = ErrorHandler.handle_error(e)
        return {"status": "degraded", "error": error_report}

# Initialize dependency injection container
container = Container()
container.wire(modules=[__name__])

def main():
    """
    Entry point for the GoldenSignalsAI application.
    This function can be used to start the application or perform initialization tasks.
    """
    logging.info("Initializing GoldenSignalsAI")
    # Add any startup logic here
    print("GoldenSignalsAI is ready to run")

if __name__ == "__main__":
    main()
