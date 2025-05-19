from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide

from agents.factory import AgentFactory
from notifications.alert_manager import AlertManager
from application.services.signal_engine import SignalEngine
from infrastructure.config_manager import config_manager
from infrastructure.data_fetcher import MarketDataFetcher

class Container(containers.DeclarativeContainer):
    """
    Dependency Injection Container for GoldenSignalsAI.
    Manages creation and lifecycle of core system components.
    """
    
    config = providers.Object(config_manager)
    
    market_data_fetcher = providers.Singleton(
        MarketDataFetcher,
        api_key=config.provided.get('alpha_vantage.api_key')
    )
    
    agent_factory = providers.Singleton(
        AgentFactory,
        data_fetcher=market_data_fetcher
    )
    
    alert_manager = providers.Singleton(
        AlertManager,
        channels={
            'email': config.provided.get('notifications.email', {}),
            'slack': config.provided.get('notifications.slack', {})
        }
    )
    
    signal_engine = providers.Singleton(
        SignalEngine,
<<<<<<< HEAD
        agent_factory=agent_factory
=======
        agent_factory=agent_factory,
        alert_manager=alert_manager
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b
    )

# Wire dependencies for easy injection
@inject
def create_signal_engine(
    signal_engine: SignalEngine = Provide[Container.signal_engine]
):
    return signal_engine

# Initialize the container
container = Container()
container.wire(modules=[
    'main',
    'orchestration.supervisor',
    'application.services.signal_engine'
])
