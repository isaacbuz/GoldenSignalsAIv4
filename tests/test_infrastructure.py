import pytest
from infrastructure.config_manager import ConfigManager
from infrastructure.dependency_container import Container
from infrastructure.error_handling import ApplicationError, ErrorSeverity

def test_config_manager():
    """Test configuration management capabilities."""
    config = ConfigManager()
    
    # Test basic configuration retrieval
    assert config.get('version') is not None
    assert config.get('environment') == 'development'
    
    # Test feature flag management
    config.set_feature_flag('test_feature', True)
    assert config.is_feature_enabled('test_feature') is True

def test_dependency_injection():
    """Test dependency injection container."""
    container = Container()
    
    # Verify core services are created
    assert container.market_data_fetcher() is not None
    assert container.agent_factory() is not None
    assert container.signal_engine() is not None

def test_error_handling():
    """Test application error handling mechanisms."""
    # Test error creation
    error = ApplicationError(
        "Test Error", 
        severity=ErrorSeverity.MEDIUM,
        context={'test_key': 'test_value'}
    )
    
    assert str(error) == "Test Error"
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.context['test_key'] == 'test_value'

def test_recovery_strategies():
    """Test error recovery strategies."""
    from governance.error_handling import default_trading_recovery, data_fetch_recovery
    
    class MockError(Exception):
        pass
    
    # Test trading recovery
    trading_recovery = default_trading_recovery(MockError("Trading error"))
    assert trading_recovery['action'] == 'hold'
    
    # Test data fetch recovery
    data_recovery = data_fetch_recovery(MockError("Data fetch error"))
    assert data_recovery['fallback_source'] == 'local_cache'
    assert data_recovery['retry_count'] == 3
