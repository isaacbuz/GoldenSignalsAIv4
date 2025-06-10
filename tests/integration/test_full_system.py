"""
Integration tests for the complete trading system.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.orchestration.orchestrator import AgentOrchestrator
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.backtesting.backtest_engine import BacktestEngine

@pytest.mark.integration
def test_full_trading_cycle(historical_data, sample_news_data):
    """Test complete trading cycle with all components"""
    # Initialize system
    orchestrator = AgentOrchestrator()
    
    # Register agents with different configurations
    agents = [
        RSIAgent(name="RSI_Fast", period=7),
        RSIAgent(name="RSI_Standard", period=14),
        MACDAgent(name="MACD_Standard"),
        MACDAgent(name="MACD_Fast", fast_period=6, slow_period=13),
        SentimentAgent(name="Sentiment_NLTK")
    ]
    
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Configure signal weights
    orchestrator.update_signal_weights({
        "technical": 0.7,
        "sentiment": 0.3
    })
    
    # Initialize backtest engine
    engine = BacktestEngine(
        orchestrator=orchestrator,
        initial_capital=100000.0,
        commission=0.001
    )
    
    # Run backtest
    results = engine.run(
        prices=historical_data["Close"],
        texts=sample_news_data * (len(historical_data) // len(sample_news_data) + 1),
        window=100
    )
    
    # Verify system integration
    assert len(results["trades"]) > 0
    assert len(results["equity_curve"]) > 0
    assert all(key in results for key in [
        "total_return", "annual_return", "sharpe_ratio",
        "max_drawdown", "win_rate", "profit_factor"
    ])

@pytest.mark.integration
def test_real_time_simulation():
    """Test system behavior in simulated real-time environment"""
    orchestrator = AgentOrchestrator()
    
    # Register agents
    for agent in [RSIAgent(), MACDAgent(), SentimentAgent()]:
        orchestrator.register_agent(agent)
    
    # Generate streaming data
    start_price = 100.0
    price_data = []
    news_data = []
    decisions = []
    
    # Simulate 100 time steps
    for i in range(100):
        # Generate price movement
        price = start_price * (1 + np.random.normal(0, 0.01))
        start_price = price
        price_data.append(price)
        
        # Generate news
        sentiment = "positive" if np.random.random() > 0.5 else "negative"
        news = f"Market sentiment is {sentiment} at time {i}"
        news_data.append(news)
        
        # Process market data
        if len(price_data) >= 30:  # Wait for enough data
            market_data = {
                "close_prices": price_data[-30:],
                "texts": news_data[-5:],
                "timestamp": datetime.now().isoformat()
            }
            decision = orchestrator.process_market_data(market_data)
            decisions.append(decision)
    
    # Verify system behavior
    assert len(decisions) > 0
    assert all(d["action"] in ["buy", "sell", "hold"] for d in decisions)
    assert all(0 <= d["confidence"] <= 1 for d in decisions)

@pytest.mark.integration
def test_system_recovery():
    """Test system recovery from errors and edge cases"""
    orchestrator = AgentOrchestrator()
    
    # Register agents
    agents = [RSIAgent(), MACDAgent(), SentimentAgent()]
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Test recovery from bad data
    bad_data_cases = [
        {"close_prices": [], "texts": []},  # Empty data
        {"close_prices": [100] * 5, "texts": []},  # Insufficient price data
        {"close_prices": "invalid", "texts": []},  # Invalid price data
        {"close_prices": [100] * 100, "texts": [1, 2, 3]},  # Invalid text data
    ]
    
    for bad_data in bad_data_cases:
        result = orchestrator.process_market_data(bad_data)
        assert result["action"] == "hold"  # System should default to hold
        assert result["confidence"] == 0.0
    
    # Verify system can recover and process valid data
    valid_data = {
        "close_prices": [100 + i for i in range(100)],
        "texts": ["Test news"] * 5,
        "timestamp": datetime.now().isoformat()
    }
    
    result = orchestrator.process_market_data(valid_data)
    assert result["action"] in ["buy", "sell", "hold"]
    assert result["confidence"] > 0

@pytest.mark.integration
def test_multi_agent_consensus():
    """Test multi-agent consensus and conflict resolution"""
    orchestrator = AgentOrchestrator()
    
    # Create agents with conflicting biases
    agents = [
        RSIAgent(name="RSI_Bullish", oversold=60),  # Bullish bias
        RSIAgent(name="RSI_Bearish", overbought=40),  # Bearish bias
        MACDAgent(name="MACD_Fast", fast_period=6),  # More sensitive
        MACDAgent(name="MACD_Slow", fast_period=24),  # Less sensitive
        SentimentAgent(name="Sentiment")
    ]
    
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Generate test data that should create conflicts
    prices = [100.0]
    for i in range(99):
        # Create oscillating price pattern
        change = 5 * np.sin(i / 5)
        prices.append(prices[-1] * (1 + change/100))
    
    # Mixed sentiment texts
    texts = [
        "Very positive outlook for growth",
        "Concerning economic indicators",
        "Strong quarterly results",
        "Market uncertainty increases"
    ]
    
    # Process data
    result = orchestrator.process_market_data({
        "close_prices": prices,
        "texts": texts,
        "timestamp": datetime.now().isoformat()
    })
    
    # Verify consensus mechanism
    assert "contributing_signals" in result
    signals = result["contributing_signals"]
    
    # Should have signals from all agents
    assert len(signals) == len(agents)
    
    # Should have some disagreement
    actions = [s["action"] for s in signals]
    assert len(set(actions)) > 1  # At least two different actions
    
    # Final confidence should reflect uncertainty
    assert result["confidence"] < 1.0 