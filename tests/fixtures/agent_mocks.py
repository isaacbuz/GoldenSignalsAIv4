"""Agent mock fixtures for testing"""

import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_base_agent():
    """Create a mock base agent"""
    agent = Mock()
    agent.name = "TestAgent"
    agent.agent_type = "test"
    agent.process = MagicMock(return_value={
        "action": "hold",
        "confidence": 0.5,
        "metadata": {}
    })
    return agent

@pytest.fixture
def mock_technical_agent():
    """Create a mock technical analysis agent"""
    agent = Mock()
    agent.name = "TechnicalAgent"
    agent.agent_type = "technical"
    agent.process = MagicMock(return_value={
        "action": "buy",
        "confidence": 0.8,
        "metadata": {
            "indicator": "RSI",
            "value": 30,
            "signal": "oversold"
        }
    })
    return agent

@pytest.fixture
def mock_sentiment_agent():
    """Create a mock sentiment analysis agent"""
    agent = Mock()
    agent.name = "SentimentAgent"
    agent.agent_type = "sentiment"
    agent.process = MagicMock(return_value={
        "action": "sell",
        "confidence": 0.7,
        "metadata": {
            "sentiment_score": -0.3,
            "news_count": 5,
            "social_mentions": 100
        }
    })
    return agent

@pytest.fixture
def mock_ml_agent():
    """Create a mock ML agent"""
    agent = Mock()
    agent.name = "MLAgent"
    agent.agent_type = "ml"
    agent.process = MagicMock(return_value={
        "action": "buy",
        "confidence": 0.9,
        "metadata": {
            "model": "RandomForest",
            "feature_importance": {"price_momentum": 0.3, "volume": 0.2},
            "prediction_probability": 0.9
        }
    })
    agent.train = MagicMock()
    agent.predict = MagicMock(return_value={"prediction": 1, "probability": 0.9})
    return agent

@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator"""
    orchestrator = Mock()
    orchestrator.agents = []
    orchestrator.process_market_data = MagicMock(return_value=[
        {"agent": "Agent1", "action": "buy", "confidence": 0.8},
        {"agent": "Agent2", "action": "hold", "confidence": 0.6},
        {"agent": "Agent3", "action": "buy", "confidence": 0.7}
    ])
    orchestrator.get_consensus = MagicMock(return_value={
        "action": "buy",
        "confidence": 0.75,
        "agreement": 0.67
    })
    return orchestrator

@pytest.fixture
def mock_market_data_service():
    """Create a mock market data service"""
    service = Mock()
    service.get_latest_data = MagicMock(return_value={
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 1000000,
        "change": 2.5,
        "change_percent": 1.69
    })
    service.get_historical_data = MagicMock(return_value=[
        {"timestamp": "2023-01-01", "close": 148},
        {"timestamp": "2023-01-02", "close": 149},
        {"timestamp": "2023-01-03", "close": 150}
    ])
    return service

@pytest.fixture
def mock_websocket_manager():
    """Create a mock WebSocket manager"""
    manager = Mock()
    manager.broadcast = MagicMock()
    manager.send_to_user = MagicMock()
    manager.connected_users = {"user1": Mock(), "user2": Mock()}
    return manager

@pytest.fixture
def mock_signal():
    """Create a mock signal object"""
    return {
        "id": "signal_123",
        "timestamp": "2023-01-01T10:00:00Z",
        "symbol": "AAPL",
        "action": "buy",
        "confidence": 0.85,
        "price": 150.0,
        "agents": ["TechnicalAgent", "MLAgent"],
        "metadata": {
            "rsi": 28,
            "macd": "bullish_crossover",
            "volume_spike": True
        }
    } 