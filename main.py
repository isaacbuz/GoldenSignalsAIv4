"""
GoldenSignalsAI V3 - Application Entry Point

This is the main entry point for the GoldenSignalsAI trading platform.
The actual FastAPI application is implemented in src/main.py
"""

import os
import sys
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import app

# Configure basic logging for the entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    try:
        import uvicorn

        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))

        # Check for command line port override
        for arg in sys.argv:
            if arg.startswith("--port="):
                try:
                    port = int(arg.split("=")[1])
                except ValueError:
                    logger.warning(f"Invalid port argument: {arg}")

        logger.info(f"🚀 Starting GoldenSignalsAI V3 on {host}:{port}")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=os.getenv("RELOAD", "false").lower() == "true",
            log_level="info"
        )

    except ImportError:
        logger.error("❌ uvicorn not installed. Please install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
