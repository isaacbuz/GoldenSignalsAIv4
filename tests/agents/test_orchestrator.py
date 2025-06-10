"""
Tests for the agent orchestrator.
"""
import pytest
from datetime import datetime
from agents.orchestration.orchestrator import AgentOrchestrator
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.sentiment.sentiment_agent import SentimentAgent

def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    orchestrator = AgentOrchestrator()
    assert len(orchestrator.agents) == 0
    assert orchestrator.signal_manager is not None
    assert not orchestrator.is_running

def test_agent_registration(rsi_agent, macd_agent, sentiment_agent):
    """Test agent registration and unregistration"""
    orchestrator = AgentOrchestrator()
    
    # Test registration
    orchestrator.register_agent(rsi_agent)
    assert len(orchestrator.agents) == 1
    assert rsi_agent.name in orchestrator.agents
    
    # Test multiple registrations
    orchestrator.register_agent(macd_agent)
    orchestrator.register_agent(sentiment_agent)
    assert len(orchestrator.agents) == 3
    
    # Test unregistration
    orchestrator.unregister_agent(rsi_agent.name)
    assert len(orchestrator.agents) == 2
    assert rsi_agent.name not in orchestrator.agents
    
    # Test unregistering non-existent agent
    orchestrator.unregister_agent("non_existent")
    assert len(orchestrator.agents) == 2

def test_market_data_processing(orchestrator, market_data):
    """Test market data processing through all agents"""
    result = orchestrator.process_market_data(market_data)
    
    assert "action" in result
    assert "confidence" in result
    assert "timestamp" in result
    assert "contributing_signals" in result
    
    assert result["action"] in ["buy", "sell", "hold"]
    assert 0 <= result["confidence"] <= 1
    assert len(result["contributing_signals"]) > 0

def test_signal_weights(orchestrator, market_data):
    """Test signal weight configuration"""
    # Test with default weights
    default_result = orchestrator.process_market_data(market_data)
    
    # Update weights to favor technical signals
    orchestrator.update_signal_weights({
        "technical": 0.8,
        "sentiment": 0.2
    })
    
    technical_result = orchestrator.process_market_data(market_data)
    
    # Results might be different due to weight changes
    assert default_result["confidence"] != technical_result["confidence"]

def test_agent_error_handling(orchestrator):
    """Test handling of agent errors"""
    # Create market data that will cause errors
    bad_data = {
        "close_prices": "invalid",  # Should be list
        "texts": [123, 456]  # Should be strings
    }
    
    # Process should complete despite errors
    result = orchestrator.process_market_data(bad_data)
    assert result["action"] == "hold"  # Default action on error
    assert result["confidence"] == 0.0

def test_agent_statistics(orchestrator, market_data):
    """Test agent statistics collection"""
    # Process some data
    orchestrator.process_market_data(market_data)
    
    # Get agent stats
    stats = orchestrator.get_agent_stats()
    
    assert len(stats) == len(orchestrator.agents)
    for stat in stats:
        assert "name" in stat
        assert "type" in stat
        assert "last_run" in stat
        assert "success_rate" in stat
        assert "total_runs" in stat

def test_signal_aggregation(orchestrator, market_data):
    """Test signal aggregation with conflicting signals"""
    # Create agents with conflicting signals
    buy_biased_rsi = RSIAgent(name="Buy_RSI", oversold=60)  # More buy signals
    sell_biased_macd = MACDAgent(name="Sell_MACD", signal_threshold=0.1)  # More sell signals
    
    # Clear existing agents and register conflicting ones
    orchestrator.agents.clear()
    orchestrator.register_agent(buy_biased_rsi)
    orchestrator.register_agent(sell_biased_macd)
    
    # Process data
    result = orchestrator.process_market_data(market_data)
    signals = result["contributing_signals"]
    
    # Verify we have conflicting signals
    signal_actions = [s["action"] for s in signals]
    assert "buy" in signal_actions or "sell" in signal_actions
    
    # Final confidence should reflect uncertainty
    assert result["confidence"] < 1.0

def test_sequential_processing(orchestrator):
    """Test processing multiple market data points"""
    data_points = []
    price = 100.0
    
    # Generate sequence of market data
    for i in range(5):
        price *= (1 + 0.01)  # 1% increase
        data_points.append({
            "close_prices": [price],
            "texts": [f"Update {i+1}: Price increased to {price:.2f}"],
            "timestamp": datetime.now().isoformat()
        })
    
    # Process sequence
    results = []
    for data in data_points:
        result = orchestrator.process_market_data(data)
        results.append(result)
    
    # Verify sequence
    assert len(results) == len(data_points)
    for result in results:
        assert "action" in result
        assert "confidence" in result
        assert "timestamp" in result 