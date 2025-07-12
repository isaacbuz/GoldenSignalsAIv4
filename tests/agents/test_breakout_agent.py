import pytest
from unittest.mock import Mock
from agents.core.technical.breakout_agent import BreakoutAgent

def test_breakout_agent_initialization():
    mock_config = Mock()
    mock_db = Mock()
    mock_redis = Mock()
    agent = BreakoutAgent(config=mock_config, db_manager=mock_db, redis_manager=mock_redis)
    assert agent is not None