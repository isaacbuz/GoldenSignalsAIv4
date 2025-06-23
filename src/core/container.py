"""
Dependency Injection Container for GoldenSignalsAI
"""

from typing import Dict, Type, Any, Optional
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Service container for dependency injection
    Manages service instances and their lifecycles
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        
    def register(self, name: str, factory: callable, singleton: bool = True):
        """Register a service factory"""
        self._factories[name] = factory
        if singleton:
            self._singletons[name] = None
        logger.info(f"Registered service: {name} (singleton={singleton})")
        
    def register_instance(self, name: str, instance: Any):
        """Register an existing instance"""
        self._services[name] = instance
        logger.info(f"Registered instance: {name}")
        
    def get(self, name: str) -> Any:
        """Get a service instance"""
        # Check if already instantiated
        if name in self._services:
            return self._services[name]
            
        # Check singletons
        if name in self._singletons:
            if self._singletons[name] is None:
                # Create singleton instance
                self._singletons[name] = self._create_instance(name)
            return self._singletons[name]
            
        # Create new instance
        if name in self._factories:
            return self._create_instance(name)
            
        raise KeyError(f"Service '{name}' not registered")
        
    def _create_instance(self, name: str) -> Any:
        """Create a new service instance"""
        factory = self._factories.get(name)
        if not factory:
            raise KeyError(f"No factory registered for '{name}'")
            
        try:
            # Pass container to factory for dependency resolution
            instance = factory(self)
            logger.debug(f"Created instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Error creating service '{name}': {e}")
            raise
            
    def reset(self):
        """Reset all services"""
        self._services.clear()
        self._singletons = {k: None for k in self._singletons}
        logger.info("Container reset")


# Global container instance
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """Get the global service container"""
    return _container


def register_services():
    """Register all application services"""
    from src.services.data_quality_validator import DataQualityValidator
    from src.services.signal_generation_engine import SignalGenerationEngine
    from src.services.signal_filtering_pipeline import SignalFilteringPipeline
    from src.services.signal_monitoring_service import SignalMonitoringService
    from src.services.market_data_manager import MarketDataManager
    from src.services.risk_manager import RiskManager
    from src.services.portfolio_simulator import PortfolioSimulator
    
    container = get_container()
    
    # Data services
    container.register("data_validator", lambda c: DataQualityValidator())
    container.register("market_data", lambda c: MarketDataManager())
    
    # Signal services
    container.register("signal_engine", lambda c: SignalGenerationEngine())
    container.register("signal_filter", lambda c: SignalFilteringPipeline())
    container.register("signal_monitor", lambda c: SignalMonitoringService())
    
    # Risk and portfolio services
    container.register("risk_manager", lambda c: RiskManager())
    container.register("portfolio", lambda c: PortfolioSimulator())
    
    logger.info("All services registered")


@lru_cache(maxsize=1)
def initialize_container():
    """Initialize the container with all services"""
    register_services()
    return get_container() 