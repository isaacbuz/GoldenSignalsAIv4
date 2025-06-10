"""
Example of using the agent-based trading system.
"""
from agents.orchestration.orchestrator import AgentOrchestrator
from agents.technical.rsi_agent import RSIAgent
from typing import Dict, Any
import time

def simulate_market_data() -> Dict[str, Any]:
    """Simulate some market data for testing"""
    import numpy as np
    
    # Generate some fake price data
    n_points = 100
    base_price = 100
    prices = [base_price]
    
    for _ in range(n_points - 1):
        change = np.random.normal(0, 1)
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)
    
    return {
        "symbol": "TEST",
        "close_prices": prices,
        "timestamp": time.time()
    }

def main():
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Create and register agents
    rsi_agent = RSIAgent(
        name="RSI_14",
        period=14,
        overbought=70,
        oversold=30
    )
    orchestrator.register_agent(rsi_agent)
    
    # Simulate market data and process
    market_data = simulate_market_data()
    decision = orchestrator.process_market_data(market_data)
    
    # Print results
    print("\nTrading Decision:")
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print("\nContributing Signals:")
    for signal in decision['contributing_signals']:
        print(f"- {signal['agent']}: {signal['action']} ({signal['confidence']:.2f})")
    
    # Print agent stats
    print("\nAgent Statistics:")
    for stats in orchestrator.get_agent_stats():
        print(f"- {stats['name']}: {stats['success_rate']:.2%} success rate")

if __name__ == "__main__":
    main() 