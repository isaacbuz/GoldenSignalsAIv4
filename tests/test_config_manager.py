import pytest
from src.infrastructure.config_manager import ConfigManager

def test_load_config_success():
    cm = ConfigManager('config/config.yaml')
    config = cm.load()
    assert isinstance(config, dict)
    assert 'api_keys' in config or 'database' in config

def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        ConfigManager('config/missing.yaml').load()
