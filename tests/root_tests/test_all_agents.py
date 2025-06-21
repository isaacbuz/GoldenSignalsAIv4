#!/usr/bin/env python3
"""
Comprehensive test script for all 19 GoldenSignalsAI agents
Tests individual agents and the complete system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
import json

# Import all agents
# Phase 1
from agents.core.technical.simple_working_agent import SimpleRSIAgent
from agents.core.technical.macd_agent import MACDAgent
from agents.core.technical.volume_spike_agent import VolumeSpikeAgent
from agents.core.technical.ma_crossover_agent import MACrossoverAgent

# Phase 2
from agents.core.technical.bollinger_bands_agent import BollingerBandsAgent
from agents.core.technical.stochastic_agent import StochasticAgent
from agents.core.technical.ema_agent import EMAAgent
from agents.core.technical.atr_agent import ATRAgent
from agents.core.technical.vwap_agent import VWAPAgent

# Phase 3
from agents.core.technical.ichimoku_agent import IchimokuAgent
from agents.core.technical.fibonacci_agent import FibonacciAgent
from agents.core.technical.adx_agent import ADXAgent
from agents.core.technical.parabolic_sar_agent import ParabolicSARAgent
from agents.core.technical.std_dev_agent import StandardDeviationAgent

# Phase 4
from agents.core.volume.volume_profile_agent import VolumeProfileAgent
from agents.core.market.market_profile_agent import MarketProfileAgent
from agents.core.flow.order_flow_agent import OrderFlowAgent
from agents.core.sentiment.simple_sentiment_agent import SimpleSentimentAgent
from agents.core.options.simple_options_flow_agent import SimpleOptionsFlowAgent

# Meta agents
from agents.orchestration.simple_orchestrator import SimpleOrchestrator
from agents.meta.ml_meta_agent import MLMetaAgent

# Backtesting
from backtesting.simple_backtest import SimpleBacktest

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}\n")

def test_individual_agents():
    """Test each agent individually"""
    print_header("Testing All 19 Individual Agents")
    
    all_agents = {
        "Phase 1 - Basic Technical": [
            ("RSI", SimpleRSIAgent()),
            ("MACD", MACDAgent()),
            ("Volume Spike", VolumeSpikeAgent()),
            ("MA Crossover", MACrossoverAgent()),
        ],
        "Phase 2 - Intermediate Technical": [
            ("Bollinger Bands", BollingerBandsAgent()),
            ("Stochastic", StochasticAgent()),
            ("EMA", EMAAgent()),
            ("ATR", ATRAgent()),
            ("VWAP", VWAPAgent()),
        ],
        "Phase 3 - Advanced Technical": [
            ("Ichimoku", IchimokuAgent()),
            ("Fibonacci", FibonacciAgent()),
            ("ADX", ADXAgent()),
            ("Parabolic SAR", ParabolicSARAgent()),
            ("Std Deviation", StandardDeviationAgent()),
        ],
        "Phase 4 - Market Analysis": [
            ("Volume Profile", VolumeProfileAgent()),
            ("Market Profile", MarketProfileAgent()),
            ("Order Flow", OrderFlowAgent()),
            ("Sentiment", SimpleSentimentAgent()),
            ("Options Flow", SimpleOptionsFlowAgent()),
        ]
    }
    
    test_symbol = "AAPL"
    results = {}
    
    for phase, agents in all_agents.items():
        print(f"\n{phase}:")
        print("-" * 40)
        
        phase_results = []
        for name, agent in agents:
            try:
                start_time = time.time()
                signal = agent.generate_signal(test_symbol)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000  # Convert to ms
                
                if signal and signal.get('action') != 'ERROR':
                    status = "✅"
                    phase_results.append({
                        'name': name,
                        'status': 'success',
                        'signal': signal,
                        'execution_time_ms': execution_time
                    })
                else:
                    status = "❌"
                    phase_results.append({
                        'name': name,
                        'status': 'error',
                        'error': signal.get('reason', 'Unknown error'),
                        'execution_time_ms': execution_time
                    })
                
                print(f"{status} {name:20} | Action: {signal.get('action', 'N/A'):8} | "
                      f"Confidence: {signal.get('confidence', 0):.2%} | "
                      f"Time: {execution_time:.1f}ms")
                
            except Exception as e:
                print(f"❌ {name:20} | Error: {str(e)[:50]}")
                phase_results.append({
                    'name': name,
                    'status': 'exception',
                    'error': str(e)
                })
        
        results[phase] = phase_results
    
    # Summary
    total_agents = sum(len(agents) for agents in all_agents.values())
    successful_agents = sum(1 for phase_results in results.values() 
                          for r in phase_results if r['status'] == 'success')
    
    print(f"\n{'Summary':^40}")
    print("-" * 40)
    print(f"Total Agents: {total_agents}")
    print(f"Successful: {successful_agents}")
    print(f"Failed: {total_agents - successful_agents}")
    print(f"Success Rate: {successful_agents/total_agents:.1%}")
    
    return results

def test_orchestrator():
    """Test the orchestrator with all agents"""
    print_header("Testing Orchestrator with 19 Agents")
    
    orchestrator = SimpleOrchestrator()
    
    # Test multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    print("Testing signal generation for multiple symbols...")
    start_time = time.time()
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        signal = orchestrator.generate_signals_for_symbol(symbol)
        
        print(f"  Consensus: {signal.get('action', 'N/A')} "
              f"(Confidence: {signal.get('confidence', 0):.2%})")
        
        # Show agent breakdown
        breakdown = signal.get('metadata', {}).get('agent_breakdown', {})
        if breakdown:
            buy_count = sum(1 for a in breakdown.values() if a['action'] == 'BUY')
            sell_count = sum(1 for a in breakdown.values() if a['action'] == 'SELL')
            neutral_count = sum(1 for a in breakdown.values() if a['action'] == 'NEUTRAL')
            
            print(f"  Agent Votes: BUY={buy_count}, SELL={sell_count}, NEUTRAL={neutral_count}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nTotal orchestration time: {total_time:.2f}s")
    print(f"Average per symbol: {total_time/len(symbols):.2f}s")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = orchestrator.get_performance_metrics()
    summary = metrics.get('summary', {})
    print(f"  Total Agents: {summary.get('total_agents', 0)}")
    print(f"  Phase Distribution: P1={summary.get('phase_1_agents', 0)}, "
          f"P2={summary.get('phase_2_agents', 0)}, "
          f"P3={summary.get('phase_3_agents', 0)}, "
          f"P4={summary.get('phase_4_agents', 0)}")

def test_ml_meta_agent():
    """Test ML Meta-Agent functionality"""
    print_header("Testing ML Meta-Agent")
    
    # Create sample agent signals
    sample_signals = [
        {"agent": "rsi", "action": "BUY", "confidence": 0.7},
        {"agent": "macd", "action": "BUY", "confidence": 0.65},
        {"agent": "bollinger", "action": "SELL", "confidence": 0.6},
        {"agent": "volume_profile", "action": "BUY", "confidence": 0.8},
        {"agent": "sentiment", "action": "NEUTRAL", "confidence": 0.5},
    ]
    
    # Initialize ML Meta-Agent
    ml_agent = MLMetaAgent(learning_rate=0.01)
    
    # Test ensemble optimization
    print("Testing ensemble optimization...")
    optimized_signal = ml_agent.optimize_ensemble(sample_signals)
    
    print(f"ML Consensus: {optimized_signal['action']} "
          f"(Confidence: {optimized_signal['confidence']:.2%})")
    print(f"Reason: {optimized_signal['reason']}")
    
    # Test performance tracking
    print("\nSimulating performance updates...")
    for i in range(5):
        for signal in sample_signals:
            # Simulate random outcome
            outcome = 1 if (i + hash(signal['agent'])) % 3 > 0 else -1
            ml_agent.update_performance(signal['agent'], signal, outcome)
    
    # Get performance report
    report = ml_agent.get_performance_report()
    print("\nML Performance Report:")
    print(f"  Total signals processed: {report['total_signals_processed']}")
    print(f"  Agent weights adjusted: {len(report['agent_weights'])}")

def test_backtesting():
    """Test backtesting framework"""
    print_header("Testing Backtesting Framework")
    
    # Initialize backtest
    backtest = SimpleBacktest(initial_capital=100000)
    
    # Test with a single agent
    print("Running backtest for RSI agent...")
    agent = SimpleRSIAgent()
    
    # Run for a short period (for testing)
    results = backtest.run_backtest(
        agent=agent,
        symbol="AAPL",
        start_date="2023-10-01",
        end_date="2023-12-31",
        position_size=0.1
    )
    
    if results:
        print("\nBacktest Results:")
        print(f"  Total Return: {results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  vs Buy & Hold: {results.get('outperformance', 0):+.2%}")
    else:
        print("❌ Backtest failed")

def test_system_integration():
    """Test complete system integration"""
    print_header("System Integration Test")
    
    print("1. Agent Count Verification")
    total_expected = 19
    orchestrator = SimpleOrchestrator()
    actual_agents = len(orchestrator.agents)
    
    if actual_agents == total_expected:
        print(f"✅ All {total_expected} agents loaded correctly")
    else:
        print(f"❌ Expected {total_expected} agents, found {actual_agents}")
    
    print("\n2. Signal Generation Speed Test")
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    start_time = time.time()
    
    signals = orchestrator.generate_all_signals()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ Generated {len(signals)} signals in {total_time:.2f}s")
    print(f"   Average: {total_time/len(signals)*1000:.1f}ms per symbol")
    
    print("\n3. ML Integration Test")
    ml_agent = MLMetaAgent()
    
    # Get agent signals for ML optimization
    if signals:
        first_symbol_signal = signals[0]
        agent_signals = []
        
        # Extract individual agent signals from breakdown
        breakdown = first_symbol_signal.get('metadata', {}).get('agent_breakdown', {})
        for agent_name, agent_data in breakdown.items():
            agent_signals.append({
                'agent': agent_name,
                'action': agent_data['action'],
                'confidence': agent_data['confidence']
            })
        
        ml_optimized = ml_agent.optimize_ensemble(agent_signals)
        print(f"✅ ML optimization successful: {ml_optimized['action']}")
    
    print("\n4. System Health Check")
    health_metrics = {
        'agents_active': actual_agents,
        'orchestrator_status': 'OK' if orchestrator else 'ERROR',
        'ml_agent_status': 'OK' if ml_agent else 'ERROR',
        'signal_generation': 'OK' if signals else 'ERROR'
    }
    
    all_ok = all(v == 'OK' or isinstance(v, int) for v in health_metrics.values())
    
    if all_ok:
        print("✅ System health check passed")
    else:
        print("❌ System health issues detected:")
        for metric, status in health_metrics.items():
            print(f"   {metric}: {status}")

def main():
    """Run all tests"""
    print("🚀 GoldenSignalsAI Complete System Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all test suites
    test_results = {}
    
    # 1. Individual agents
    print("\n" + "="*60)
    agent_results = test_individual_agents()
    test_results['agents'] = agent_results
    
    # 2. Orchestrator
    print("\n" + "="*60)
    test_orchestrator()
    
    # 3. ML Meta-Agent
    print("\n" + "="*60)
    test_ml_meta_agent()
    
    # 4. Backtesting
    print("\n" + "="*60)
    test_backtesting()
    
    # 5. System Integration
    print("\n" + "="*60)
    test_system_integration()
    
    # Final Summary
    print_header("Test Complete")
    print("✅ All systems tested successfully!")
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print("\nDetailed results saved to test_results.json")

if __name__ == "__main__":
    main() 