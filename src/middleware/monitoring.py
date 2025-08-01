"""
Monitoring Middleware for GoldenSignalsAI V3

Handles request monitoring, metrics collection, and performance tracking.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Monitoring middleware for request tracking and performance metrics.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and track monitoring metrics.
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log request start
        logger.info(
            f"Request started - ID: {request_id}, Method: {request.method}, "
            f"URL: {request.url}, Client: {request.client.host if request.client else 'unknown'}"
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add monitoring headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{process_time:.4f}s"

            # Log request completion
            logger.info(
                f"Request completed - ID: {request_id}, Status: {response.status_code}, "
                f"Time: {process_time:.4f}s"
            )

            return response

        except Exception as e:
            # Calculate processing time for error case
            process_time = time.time() - start_time

            # Log request error
            logger.error(
                f"Request failed - ID: {request_id}, Error: {str(e)}, " f"Time: {process_time:.4f}s"
            )

            raise
