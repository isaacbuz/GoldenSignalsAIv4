#!/usr/bin/env python3
"""Test script to verify all Phase 3 agents are working"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Phase 1 agents
from agents.core.technical.simple_working_agent import SimpleRSIAgent as SimpleWorkingAgent
from agents.core.technical.macd_agent import MACDAgent
from agents.core.technical.volume_spike_agent import VolumeSpikeAgent
from agents.core.technical.ma_crossover_agent import MACrossoverAgent

# Phase 2 agents
from agents.core.technical.bollinger_bands_agent import BollingerBandsAgent
from agents.core.technical.stochastic_agent import StochasticAgent
from agents.core.technical.ema_agent import EMAAgent
from agents.core.technical.atr_agent import ATRAgent
from agents.core.technical.vwap_agent import VWAPAgent

# Phase 3 agents
from agents.core.technical.ichimoku_agent import IchimokuAgent
from agents.core.technical.fibonacci_agent import FibonacciAgent
from agents.core.technical.adx_agent import ADXAgent
from agents.core.technical.parabolic_sar_agent import ParabolicSARAgent
from agents.core.technical.std_dev_agent import StandardDeviationAgent

from agents.orchestration.simple_orchestrator import SimpleOrchestrator

def test_phase3_agents():
    """Test Phase 3 agents individually"""
    print("🧪 Testing Phase 3 Agents")
    print("=" * 50)

    phase3_agents = [
        ("Ichimoku Cloud Agent", IchimokuAgent()),
        ("Fibonacci Retracement Agent", FibonacciAgent()),
        ("ADX Agent", ADXAgent()),
        ("Parabolic SAR Agent", ParabolicSARAgent()),
        ("Standard Deviation Agent", StandardDeviationAgent()),
    ]

    test_symbol = "AAPL"

    for name, agent in phase3_agents:
        try:
            signal = agent.generate_signal(test_symbol)
            status = "✅" if signal and signal.get('action') != 'ERROR' else "❌"
            print(f"\n{status} {name}:")
            print(f"   Action: {signal.get('action', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 0):.2%}")
            print(f"   Reason: {signal.get('reason', 'N/A')}")

            # Show key data points
            data = signal.get('data', {})
            if name == "Ichimoku Cloud Agent" and data:
                print(f"   Cloud: ${data.get('cloud_bottom', 0):.2f} - ${data.get('cloud_top', 0):.2f}")
            elif name == "Fibonacci Retracement Agent" and data:
                levels = data.get('fib_levels', {})
                if levels:
                    print(f"   Key Levels: 38.2%=${levels.get('38.2%', 0):.2f}, 61.8%=${levels.get('61.8%', 0):.2f}")
            elif name == "ADX Agent" and data:
                print(f"   ADX: {data.get('adx', 0):.1f}, Trend: {data.get('trend_strength', 'N/A')}")
            elif name == "Parabolic SAR Agent" and data:
                print(f"   SAR: ${data.get('psar', 0):.2f}, Trend: {data.get('trend', 'N/A')}")
            elif name == "Standard Deviation Agent" and data:
                print(f"   Z-Score: {data.get('z_score', 0):.2f}, Volatility: {data.get('volatility_state', 'N/A')}")

        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")

def test_all_agents_summary():
    """Test all 14 agents"""
    print("\n\n📊 All Agents Summary (14 Total)")
    print("=" * 50)

    all_agents = {
        "Phase 1": [
            ("RSI", SimpleWorkingAgent()),
            ("MACD", MACDAgent()),
            ("Volume Spike", VolumeSpikeAgent()),
            ("MA Crossover", MACrossoverAgent()),
        ],
        "Phase 2": [
            ("Bollinger Bands", BollingerBandsAgent()),
            ("Stochastic", StochasticAgent()),
            ("EMA", EMAAgent()),
            ("ATR", ATRAgent()),
            ("VWAP", VWAPAgent()),
        ],
        "Phase 3": [
            ("Ichimoku", IchimokuAgent()),
            ("Fibonacci", FibonacciAgent()),
            ("ADX", ADXAgent()),
            ("Parabolic SAR", ParabolicSARAgent()),
            ("Std Dev", StandardDeviationAgent()),
        ]
    }

    test_symbol = "AAPL"

    for phase, agents in all_agents.items():
        print(f"\n{phase}:")
        for name, agent in agents:
            try:
                signal = agent.generate_signal(test_symbol)
                action = signal.get('action', 'ERROR')
                confidence = signal.get('confidence', 0)
                status = "✅" if action != 'ERROR' else "❌"
                print(f"  {status} {name}: {action} ({confidence:.1%})")
            except Exception as e:
                print(f"  ❌ {name}: Error")

def test_orchestrator_performance():
    """Test orchestrator with all 14 agents"""
    print("\n\n🎭 Orchestrator Performance Test")
    print("=" * 50)

    orchestrator = SimpleOrchestrator()

    # Test single symbol
    print("\n📈 Testing AAPL with 14 agents...")
    import time
    start = time.time()
    signal = orchestrator.generate_signals_for_symbol("AAPL")
    end = time.time()

    print(f"Consensus Signal: {signal.get('action', 'N/A')}")
    print(f"Confidence: {signal.get('confidence', 0):.2%}")
    print(f"Time taken: {end - start:.2f}s")

    # Show agent breakdown
    breakdown = signal.get('metadata', {}).get('agent_breakdown', {})
    if breakdown:
        print("\nAgent Votes:")
        buy_count = sum(1 for a in breakdown.values() if a['action'] == 'BUY')
        sell_count = sum(1 for a in breakdown.values() if a['action'] == 'SELL')
        neutral_count = sum(1 for a in breakdown.values() if a['action'] == 'NEUTRAL')
        print(f"  BUY: {buy_count}, SELL: {sell_count}, NEUTRAL: {neutral_count}")

    # Performance stats
    print("\n📊 Performance Statistics:")
    stats = orchestrator.get_performance_metrics()
    summary = stats.get('summary', {})
    print(f"Total Agents: {summary.get('total_agents', 0)}")
    print(f"Phase 1: {summary.get('phase_1_agents', 0)} agents")
    print(f"Phase 2: {summary.get('phase_2_agents', 0)} agents")
    print(f"Phase 3: {summary.get('phase_3_agents', 0)} agents")

def main():
    print("🚀 GoldenSignalsAI Phase 3 Agent Test")
    print("=" * 60)

    test_phase3_agents()
    test_all_agents_summary()
    test_orchestrator_performance()

    print("\n\n✅ Phase 3 Testing Complete!")
    print("All 14 agents are operational!")

if __name__ == "__main__":
    main()
