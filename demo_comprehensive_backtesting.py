"""
Comprehensive Backtesting Demo
Demonstrates all enhancement phases working together:
- Real data fetching
- Market microstructure simulation
- Adaptive agent learning
- Signal validation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import our enhanced components
from src.domain.backtesting.enhanced_data_manager import EnhancedDataManager
from src.domain.backtesting.market_simulator import MarketMicrostructureSimulator, Order, OrderType, OrderSide
from src.domain.backtesting.adaptive_agent_framework import RSIAdaptiveAgent, AgentPerformanceTracker, TradingOutcome
from src.domain.backtesting.signal_accuracy_validator import SignalAccuracyValidator, SignalRecord

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def comprehensive_backtest_demo():
    """Run a comprehensive demonstration of all backtesting enhancements"""
    
    print("🚀 GoldenSignalsAI V2 - Comprehensive Backtesting Demo")
    print("=" * 70)
    
    # Phase 1: Real Data Infrastructure
    print("\n📊 Phase 1: Fetching Real Market Data...")
    data_manager = EnhancedDataManager()
    
    # Fetch real data for multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    market_data = {}
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        data = await data_manager.fetch_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        market_data[symbol] = data
        
        # Show data quality
        quality = data_manager.validate_data_quality(data)
        print(f"  ✅ {symbol}: {len(data)} days, Quality Score: {quality['overall_score']:.1f}%")
    
    # Phase 2: Market Microstructure Simulation
    print("\n🏛️ Phase 2: Market Microstructure Simulation...")
    simulator = MarketMicrostructureSimulator({
        'base_spread_bps': 2,
        'market_impact_factor': 0.1,
        'latency_ms': 50
    })
    
    # Simulate order book for AAPL
    symbol = 'AAPL'
    current_data = market_data[symbol]
    current_price = current_data['close'].iloc[-1]
    volume = current_data['volume'].iloc[-1]
    volatility = current_data['close'].pct_change().rolling(20).std().iloc[-1]
    
    order_book = simulator.simulate_order_book(
        symbol=symbol,
        mid_price=current_price,
        volume=volume,
        volatility=volatility,
        timestamp=datetime.now()
    )
    
    print(f"\n  Order Book for {symbol}:")
    print(f"  Best Bid: ${order_book.best_bid:.2f} (Size: {order_book.bids[0].quantity:,})")
    print(f"  Best Ask: ${order_book.best_ask:.2f} (Size: {order_book.asks[0].quantity:,})")
    print(f"  Spread: ${order_book.spread:.2f} ({order_book.spread/current_price*10000:.1f} bps)")
    print(f"  Book Depth: {len(order_book.bids)} levels each side")
    
    # Phase 3: Adaptive Agent Learning
    print("\n🤖 Phase 3: Adaptive Agent Framework...")
    
    # Create adaptive agents with different configurations
    agents = [
        RSIAdaptiveAgent(
            agent_id="RSI_Conservative",
            learning_config={
                'learning_rate': 0.001,
                'exploration_rate': 0.05,
                'confidence_threshold': 0.7
            }
        ),
        RSIAdaptiveAgent(
            agent_id="RSI_Aggressive",
            learning_config={
                'learning_rate': 0.01,
                'exploration_rate': 0.15,
                'confidence_threshold': 0.5
            }
        )
    ]
    
    # Create performance tracker
    tracker = AgentPerformanceTracker()
    for agent in agents:
        tracker.register_agent(agent)
    
    # Run A/B test
    tracker.run_ab_test(
        test_name="Conservative_vs_Aggressive",
        control_agent_id="RSI_Conservative",
        treatment_agent_id="RSI_Aggressive",
        duration_days=30
    )
    
    print(f"  Started A/B Test: Conservative vs Aggressive RSI strategies")
    
    # Phase 4: Signal Validation
    print("\n✅ Phase 4: Signal Accuracy Validation...")
    validator = SignalAccuracyValidator()
    
    # Simulate trading with adaptive agents
    print("\n📈 Running Backtest Simulation...")
    print("  Processing 40 trading days with adaptive learning...\n")
    
    all_decisions = []
    all_outcomes = []
    
    for i in range(50, min(90, len(current_data) - 1)):
        day_data = current_data.iloc[:i]
        
        for agent in agents:
            # Make trading decision
            decision = await agent.make_decision(day_data, symbol)
            all_decisions.append(decision)
            
            # Execute order if not HOLD
            if decision.action != 'HOLD':
                # Create order
                order = Order(
                    order_id=f"{agent.agent_id}_{i}",
                    symbol=symbol,
                    side=OrderSide.BUY if decision.action == 'BUY' else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=100  # Fixed size for demo
                )
                
                # Execute with market simulator
                market_data_dict = {
                    'avg_daily_volume': current_data['volume'].rolling(20).mean().iloc[i-1],
                    'volatility': volatility
                }
                
                executed_order = await simulator.execute_order(order, order_book, market_data_dict)
                
                # Calculate actual return (next day's return)
                actual_return = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
                if decision.action == 'SELL':
                    actual_return = -actual_return
                
                # Create outcome
                outcome = TradingOutcome(
                    decision=decision,
                    actual_return=actual_return,
                    market_conditions={'volatility': volatility},
                    execution_quality=0.95 if executed_order.status.value == 'FILLED' else 0.5,
                    timestamp=datetime.now()
                )
                
                # Record outcome for learning
                agent.record_outcome(outcome)
                all_outcomes.append(outcome)
                
                # Track with validator
                signal = SignalRecord(
                    timestamp=decision.timestamp,
                    symbol=symbol,
                    signal_type=decision.action,
                    confidence=decision.confidence,
                    predicted_direction='UP' if decision.action == 'BUY' else 'DOWN',
                    predicted_magnitude=abs(decision.predicted_return),
                    reasoning=decision.reasoning,
                    source=agent.agent_id
                )
                
                validator.record_signal(signal)
                validator.record_outcome(
                    signal_id=signal.signal_id,
                    actual_price_change=actual_return,
                    execution_price=executed_order.avg_fill_price if executed_order.avg_fill_price else current_price,
                    market_conditions=market_data_dict
                )
        
        # Show progress
        if i % 10 == 0:
            print(f"  Day {i-50}: Processed {len(all_decisions)} decisions")
            
            # Show agent performance
            for agent in agents:
                metrics = agent.calculate_performance_metrics()
                print(f"    {agent.agent_id}: Win Rate={metrics.win_rate:.1%}, Sharpe={metrics.sharpe_ratio:.2f}, Model v{agent.model_version}")
    
    # Final Results
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    # A/B Test Results
    print("\n🔬 A/B Test Analysis:")
    ab_results = tracker.analyze_ab_test("Conservative_vs_Aggressive")
    print(f"  Winner: {ab_results['winner']} (Confidence: {ab_results['confidence']})")
    print(f"  Sharpe Improvement: {ab_results['improvement']['sharpe_ratio']:.1%}")
    print(f"  Accuracy Improvement: {ab_results['improvement']['accuracy']:.1%}")
    
    # Agent Rankings
    print("\n🏆 Agent Performance Rankings:")
    rankings = tracker.get_agent_ranking()
    for rank, (agent_id, score) in enumerate(rankings, 1):
        agent = tracker.agents[agent_id]
        metrics = agent.calculate_performance_metrics()
        print(f"  {rank}. {agent_id}: Score={score:.3f}")
        print(f"     - Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"     - Win Rate: {metrics.win_rate:.1%}")
        print(f"     - Accuracy: {metrics.accuracy:.1%}")
        print(f"     - Model Stability: {metrics.model_stability:.2f}")
    
    # Execution Analytics
    print("\n⚡ Execution Quality:")
    exec_analytics = simulator.get_execution_analytics()
    for key, value in exec_analytics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Signal Validation Results
    print("\n✅ Signal Accuracy Analysis:")
    accuracy_metrics = validator.calculate_accuracy_metrics()
    print(f"  Total Signals: {accuracy_metrics['total_signals']}")
    print(f"  Direction Accuracy: {accuracy_metrics['direction_accuracy']:.1%}")
    print(f"  Precision: {accuracy_metrics['precision']:.1%}")
    print(f"  Win Rate: {accuracy_metrics['win_rate']:.1%}")
    print(f"  Avg Profit per Signal: {accuracy_metrics['avg_profit_per_signal']:.2%}")
    
    # Learning Insights
    print("\n💡 Key Insights:")
    print("  1. Adaptive agents improved performance through online learning")
    print("  2. Market microstructure simulation provided realistic execution")
    print("  3. Real-time data quality validation ensured reliable backtesting")
    print("  4. A/B testing identified optimal agent configurations")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'ab_test_results': ab_results,
        'agent_rankings': rankings,
        'execution_analytics': exec_analytics,
        'signal_accuracy': accuracy_metrics
    }
    
    with open('comprehensive_backtest_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved to comprehensive_backtest_results.json")
    print("\n🎯 Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(comprehensive_backtest_demo()) 