# run_services.py
# Purpose: Main entry point to start all services for GoldenSignalsAI, including
# the FastAPI backend, React frontend, and Dash dashboard for options trading.

import logging
import subprocess

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def start_services():
    """Start all services for GoldenSignalsAI."""
    logger.info({"message": "Starting GoldenSignalsAI services"})
    try:
        # Start Redis server
        subprocess.Popen(
            ["redis-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info({"message": "Redis server started"})

        # Start FastAPI backend
        subprocess.Popen(
            [
                "uvicorn",
                "presentation.api.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info({"message": "FastAPI backend started on port 8000"})

        # Start React frontend
        subprocess.Popen(
            ["npm", "start"],
            cwd="presentation/frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,  # shell=True for npm command on Windows/Mac
        )
        logger.info({"message": "React frontend started"})

        # Start Dash dashboard
        subprocess.Popen(
            ["python", "-m", "monitoring.agent_dashboard"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info({"message": "Dash dashboard started on port 8050"})

    except Exception as e:
        logger.error({"message": f"Failed to start services: {str(e)}"})
        raise


if __name__ == "__main__":
    start_services()
