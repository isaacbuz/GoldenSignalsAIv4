"""
Performance tests for the trading system.
"""
import pytest
import time
import psutil
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from agents.orchestration.orchestrator import AgentOrchestrator
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.backtesting.backtest_engine import BacktestEngine

def measure_execution_time(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

@pytest.mark.performance
def test_processing_latency():
    """Test processing latency for market data analysis"""
    orchestrator = AgentOrchestrator()
    
    # Register standard agents
    for agent in [RSIAgent(), MACDAgent(), SentimentAgent()]:
        orchestrator.register_agent(agent)
    
    # Prepare test data
    prices = [100.0 * (1 + np.random.normal(0, 0.01)) for _ in range(100)]
    texts = ["Market update " + str(i) for i in range(5)]
    market_data = {
        "close_prices": prices,
        "texts": texts,
        "timestamp": datetime.now().isoformat()
    }
    
    # Measure processing time
    latencies = []
    for _ in range(100):  # Run 100 times
        start_time = time.time()
        orchestrator.process_market_data(market_data)
        latency = time.time() - start_time
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Assert performance requirements
    assert avg_latency < 0.1  # Average latency under 100ms
    assert p95_latency < 0.2  # 95th percentile under 200ms
    assert p99_latency < 0.5  # 99th percentile under 500ms

@pytest.mark.performance
def test_system_throughput():
    """Test system throughput under load"""
    orchestrator = AgentOrchestrator()
    for agent in [RSIAgent(), MACDAgent(), SentimentAgent()]:
        orchestrator.register_agent(agent)
    
    def process_batch(batch_id):
        prices = [100.0 * (1 + np.random.normal(0, 0.01)) for _ in range(100)]
        texts = [f"Update {batch_id}_{i}" for i in range(5)]
        market_data = {
            "close_prices": prices,
            "texts": texts,
            "timestamp": datetime.now().isoformat()
        }
        return orchestrator.process_market_data(market_data)
    
    # Test concurrent processing
    num_requests = 100
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_batch, range(num_requests)))
    
    end_time = time.time()
    duration = end_time - start_time
    throughput = num_requests / duration
    
    # Assert minimum throughput
    assert throughput >= 10  # At least 10 requests per second
    assert all(r["action"] in ["buy", "sell", "hold"] for r in results)

@pytest.mark.performance
def test_memory_usage():
    """Test memory usage under continuous operation"""
    orchestrator = AgentOrchestrator()
    for agent in [RSIAgent(), MACDAgent(), SentimentAgent()]:
        orchestrator.register_agent(agent)
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run continuous processing
    for _ in range(1000):
        prices = [100.0 * (1 + np.random.normal(0, 0.01)) for _ in range(100)]
        texts = ["Market update"] * 5
        market_data = {
            "close_prices": prices,
            "texts": texts,
            "timestamp": datetime.now().isoformat()
        }
        orchestrator.process_market_data(market_data)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = final_memory - initial_memory
    
    # Assert reasonable memory growth
    assert memory_growth < 100  # Less than 100MB growth

@pytest.mark.performance
def test_backtest_performance():
    """Test backtesting performance with large datasets"""
    orchestrator = AgentOrchestrator()
    for agent in [RSIAgent(), MACDAgent(), SentimentAgent()]:
        orchestrator.register_agent(agent)
    
    engine = BacktestEngine(
        orchestrator=orchestrator,
        initial_capital=100000.0,
        commission=0.001
    )
    
    # Generate large dataset
    num_days = 1000
    prices = pd.Series([100.0 * (1 + np.random.normal(0, 0.01)) 
                       for _ in range(num_days)])
    texts = ["Daily update"] * num_days
    
    # Measure backtesting time
    start_time = time.time()
    results = engine.run(prices, texts, window=100)
    execution_time = time.time() - start_time
    
    # Assert performance metrics
    assert execution_time < 60  # Complete within 60 seconds
    assert len(results["equity_curve"]) == len(prices)
    assert "total_return" in results

@pytest.mark.performance
def test_agent_scaling():
    """Test system performance with increasing number of agents"""
    base_agents = [RSIAgent(), MACDAgent(), SentimentAgent()]
    
    latencies = []
    agent_counts = []
    
    # Test with increasing number of agents
    for num_duplicates in [1, 2, 5, 10]:
        orchestrator = AgentOrchestrator()
        
        # Register multiple copies of each agent
        for _ in range(num_duplicates):
            for agent_class in [RSIAgent, MACDAgent, SentimentAgent]:
                agent = agent_class(name=f"{agent_class.__name__}_{_}")
                orchestrator.register_agent(agent)
        
        # Prepare test data
        market_data = {
            "close_prices": [100.0 * (1 + np.random.normal(0, 0.01)) 
                           for _ in range(100)],
            "texts": ["Test update"] * 5,
            "timestamp": datetime.now().isoformat()
        }
        
        # Measure processing time
        start_time = time.time()
        orchestrator.process_market_data(market_data)
        latency = time.time() - start_time
        
        latencies.append(latency)
        agent_counts.append(len(orchestrator.agents))
    
    # Verify sub-linear scaling
    # Latency should not increase linearly with agent count
    latency_ratios = [latencies[i+1]/latencies[i] 
                     for i in range(len(latencies)-1)]
    agent_ratios = [agent_counts[i+1]/agent_counts[i] 
                   for i in range(len(agent_counts)-1)]
    
    # Assert sub-linear scaling
    assert all(l_ratio < a_ratio for l_ratio, a_ratio 
              in zip(latency_ratios, agent_ratios)) 