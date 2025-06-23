"""
Dependency injection container for the application
"""

from dependency_injector import containers, providers
from src.repositories.market.market_repository import MarketRepository
from src.repositories.signals.signal_repository import SignalRepository
from src.services.signals.signal_service import SignalService
from src.services.market.data_service import MarketDataService
from src.services.market.quality_validator import DataQualityValidator

class Container(containers.DeclarativeContainer):
    """Application DI container"""
    
    # Configuration
    config = providers.Configuration()
    
    # Repositories
    market_repository = providers.Singleton(
        MarketRepository,
        api_key=config.market.api_key,
        cache_enabled=config.cache.enabled
    )
    
    signal_repository = providers.Singleton(
        SignalRepository,
        database_url=config.database.url
    )
    
    # Services
    quality_validator = providers.Singleton(
        DataQualityValidator
    )
    
    market_service = providers.Singleton(
        MarketDataService,
        repository=market_repository,
        validator=quality_validator
    )
    
    signal_service = providers.Singleton(
        SignalService,
        market_service=market_service,
        signal_repository=signal_repository,
        ml_models=providers.Dict({
            'default': providers.Singleton(lambda: load_ml_model('default')),
            'advanced': providers.Singleton(lambda: load_ml_model('advanced'))
        })
    )

def load_ml_model(model_type: str):
    """Load ML model by type"""
    # Implementation to load models
    pass
