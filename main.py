<<<<<<< HEAD
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

def main() -> None:
    """Bootstrap and run the GoldenSignalsAI FastAPI server."""
    logging.info("🟢 Starting GoldenSignalsAI")
    # TODO: preload ML models, warm caches, initialize DB connections
    host = config_manager.get("HOST", "0.0.0.0")
    port = int(config_manager.get("PORT", 8000))
    logging.info(f"Serving on http://{host}:{port}")
    try:
        import uvicorn
        uvicorn.run("main:app", host=host, port=port, log_level="info")
    except ImportError:
        logging.error("uvicorn not installed—please `pip install uvicorn`");

if __name__ == "__main__":
    main()
=======
import os
import sys
from fastapi import FastAPI
from backend.api import analyze, train, logs
from backend.api import agent_weights
from backend.api import model_info
from backend.api import agents
from backend.log_config import setup_logging

setup_logging()
app = FastAPI()

app.include_router(analyze.router)
app.include_router(train.router)
app.include_router(logs.router)
app.include_router(agent_weights.router)
app.include_router(model_info.router)
app.include_router(agents.router, prefix="/api/agents")

@app.get("/")
def read_root():
    return {"message": "GoldenSignalsAI v4 is live"}

if __name__ == "__main__":
    import uvicorn
    # Priority: command-line arg > env var > default
    port = 8000
    for arg in sys.argv:
        if arg.startswith("--port="):
            try:
                port = int(arg.split("=")[1])
            except Exception:
                pass
    port = int(os.environ.get("PORT", port))
    print(f"[GoldenSignalsAI] Launching on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
>>>>>>> a9235431 (Initial commit: Add GoldenSignalsAI_Merged_Final with ML agents, retraining automation, and advanced frontend visualization)
