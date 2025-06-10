"""
Tests for the base agent functionality.
"""
import pytest
from datetime import datetime
from agents.base.base_agent import BaseAgent

class TestAgent(BaseAgent):
    """Test agent implementation"""
    def __init__(self, name: str = "Test"):
        super().__init__(name=name, agent_type="test")
        
    def process(self, data):
        if data.get("should_fail"):
            raise ValueError("Test error")
        return {"result": "success"}

def test_agent_initialization():
    """Test agent initialization"""
    agent = TestAgent("TestAgent")
    assert agent.name == "TestAgent"
    assert agent.agent_type == "test"
    assert agent.last_run is None
    assert agent.error_count == 0
    assert agent.success_count == 0

def test_successful_processing():
    """Test successful data processing"""
    agent = TestAgent()
    result = agent.run({"data": "test"})
    
    assert result["status"] == "success"
    assert result["agent"] == "Test"
    assert result["type"] == "test"
    assert "timestamp" in result
    assert result["data"]["result"] == "success"
    assert agent.success_count == 1
    assert agent.error_count == 0
    assert isinstance(agent.last_run, datetime)

def test_failed_processing():
    """Test error handling during processing"""
    agent = TestAgent()
    result = agent.run({"should_fail": True})
    
    assert result["status"] == "error"
    assert result["agent"] == "Test"
    assert result["type"] == "test"
    assert "timestamp" in result
    assert "Test error" in result["error"]
    assert agent.success_count == 0
    assert agent.error_count == 1

def test_pre_post_processing():
    """Test pre and post processing hooks"""
    agent = TestAgent()
    
    # Test pre-processing
    data = {"test": "data"}
    processed = agent.pre_process(data)
    assert processed == data  # Default implementation returns data as is
    
    # Test post-processing
    result = {"result": "test"}
    processed = agent.post_process(result)
    assert processed == result  # Default implementation returns result as is

def test_get_stats():
    """Test agent statistics"""
    agent = TestAgent()
    
    # Initial stats
    stats = agent.get_stats()
    assert stats["name"] == "Test"
    assert stats["type"] == "test"
    assert stats["last_run"] is None
    assert stats["success_rate"] == 0
    assert stats["total_runs"] == 0
    
    # After successful run
    agent.run({"data": "test"})
    stats = agent.get_stats()
    assert stats["success_rate"] == 1.0
    assert stats["total_runs"] == 1
    assert stats["last_run"] is not None
    
    # After failed run
    agent.run({"should_fail": True})
    stats = agent.get_stats()
    assert stats["success_rate"] == 0.5
    assert stats["total_runs"] == 2 