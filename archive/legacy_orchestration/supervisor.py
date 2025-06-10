# Purpose: Coordinates agent tasks with circuit breakers for reliability.
# Manages agent lifecycle, resolves conflicts, and integrates with real-time data processing.

import asyncio
import logging
from typing import Dict, List, Any

import yaml

from agents.factory import AgentFactory
from notifications.alert_manager import AlertManager
from goldensignalsai.application.services.signal_engine import SignalEngine

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {}


class CircuitBreaker:
    """Simple circuit breaker to manage agent task failures."""

    def __init__(self, max_failures: int = 3, reset_timeout: int = 300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False

    def record_failure(self):
        """Record a task failure and update circuit breaker state."""
        self.failures += 1
        if self.failures >= self.max_failures:
            self.is_open = True
            logger.warning({"message": "Circuit breaker opened due to multiple failures"})

    def reset(self):
        """Reset the circuit breaker state."""
        self.failures = 0
        self.is_open = False
        logger.info({"message": "Circuit breaker reset"})


class AgentSupervisor:
    """Manages and coordinates multiple agents."""

    def __init__(self, agent_factory: AgentFactory, alert_manager: AlertManager):
        """
        Initialize the AgentSupervisor.

        Args:
            agent_factory (AgentFactory): Factory for creating agents.
            alert_manager (AlertManager): Manager for sending alerts.
        """
        self.agent_factory = agent_factory
        self.alert_manager = alert_manager
        self.circuit_breaker = CircuitBreaker()
        self.signal_engine = SignalEngine(agent_factory)
        logger.info({"message": "AgentSupervisor initialized"})

    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data through multiple agents.

        Args:
            market_data (Dict[str, Any]): Comprehensive market data.

        Returns:
            Dict[str, Any]: Aggregated trading decision.
        """
        try:
            # Check if circuit breaker is open
            if self.circuit_breaker.is_open:
                logger.warning({"message": "Circuit breaker is open, skipping processing"})
                return {"action": "hold", "confidence": 0.0, "metadata": {"reason": "Circuit breaker open"}}

            # Generate signals
            signals = self.signal_engine.generate_signals(market_data)

            # Evaluate signals
            decision = self.signal_engine.evaluate_signals(signals)

            # Send alerts based on decision
            if decision['action'] != 'hold':
                self.alert_manager.send_trading_alert(decision)

            return decision

        except Exception as e:
            # Record failure and log error
            self.circuit_breaker.record_failure()
            logger.error({"message": f"Market data processing failed: {str(e)}"})
            self.alert_manager.send_error_alert(str(e))
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def start(self):
        """Start the supervisor's monitoring and processing tasks."""
        logger.info({"message": "AgentSupervisor starting"})
        # Placeholder for more complex startup logic
        pass

    def stop(self):
        """Stop the supervisor's tasks."""
        logger.info({"message": "AgentSupervisor stopping"})
        # Placeholder for cleanup and shutdown logic
        pass
