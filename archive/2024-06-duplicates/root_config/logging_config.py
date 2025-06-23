import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import settings


def configure_logging():
    """
    Configure comprehensive logging for the application
    """
    # Create logger
    logger = logging.getLogger("goldensignalsai")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # File Handler
    try:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    except PermissionError:
        print(f"Warning: Unable to create log file at {settings.LOG_FILE}")
        logger.addHandler(console_handler)

    return logger


# Create a global logger instance
logger = configure_logging()
