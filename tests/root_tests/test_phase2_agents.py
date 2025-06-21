#!/usr/bin/env python3
"""Test script to verify all Phase 2 agents are working"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.core.technical.simple_working_agent import SimpleWorkingAgent
from agents.core.technical.macd_agent import MACDAgent
from agents.core.technical.volume_spike_agent import VolumeSpikeAgent
from agents.core.technical.ma_crossover_agent import MACrossoverAgent
from agents.core.technical.bollinger_bands_agent import BollingerBandsAgent
from agents.core.technical.stochastic_agent import StochasticAgent
from agents.core.technical.ema_agent import EMAAgent
from agents.core.technical.atr_agent import ATRAgent
from agents.core.technical.vwap_agent import VWAPAgent
from agents.orchestration.simple_orchestrator import SimpleOrchestrator

def test_individual_agents():
    """Test each agent individually"""
    print("🧪 Testing Individual Agents")
    print("=" * 50)
    
    agents = [
        ("RSI Agent", SimpleWorkingAgent()),
        ("MACD Agent", MACDAgent()),
        ("Volume Spike Agent", VolumeSpikeAgent()),
        ("MA Crossover Agent", MACrossoverAgent()),
        ("Bollinger Bands Agent", BollingerBandsAgent()),
        ("Stochastic Oscillator Agent", StochasticAgent()),
        ("EMA Agent", EMAAgent()),
        ("ATR Agent", ATRAgent()),
        ("VWAP Agent", VWAPAgent()),
    ]
    
    test_symbol = "AAPL"
    
    for name, agent in agents:
        try:
            signal = agent.generate_signal(test_symbol)
            status = "✅" if signal else "❌"
            print(f"{status} {name}: {signal}")
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
    
    print()

def test_orchestrator():
    """Test the orchestrator with all agents"""
    print("🎭 Testing Orchestrator")
    print("=" * 50)
    
    orchestrator = SimpleOrchestrator()
    
    # Test single symbol
    print("\n📊 Single Symbol Test (AAPL):")
    signal = orchestrator.generate_signal("AAPL")
    print(f"Consensus Signal: {signal}")
    
    # Test multiple symbols
    print("\n📊 Multiple Symbols Test:")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    signals = orchestrator.generate_signals(symbols)
    for symbol, signal in signals.items():
        print(f"{symbol}: {signal}")
    
    # Show performance stats
    print("\n📈 Performance Statistics:")
    stats = orchestrator.get_performance_stats()
    print(f"Total Signals: {stats['total_signals']}")
    print(f"Average Time: {stats['avg_time']:.3f}s")
    print(f"Agent Performance:")
    for agent, perf in stats['agent_performance'].items():
        print(f"  - {agent}: {perf['success_rate']:.1%} success rate")

def main():
    print("🚀 GoldenSignalsAI Phase 2 Agent Test")
    print("=" * 50)
    
    test_individual_agents()
    test_orchestrator()
    
    print("\n✅ Phase 2 Testing Complete!")

if __name__ == "__main__":
    main() 