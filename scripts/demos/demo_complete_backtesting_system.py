"""
Complete Backtesting System Demo
Demonstrates the fully integrated backtesting system with all enhancements
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List
import matplotlib.pyplot as plt

# Import all our enhanced components
from src.domain.backtesting.enhanced_data_manager import EnhancedDataManager
from src.domain.backtesting.market_simulator import MarketMicrostructureSimulator, Order, OrderType, OrderSide
from src.domain.backtesting.adaptive_agent_framework import RSIAdaptiveAgent, AgentPerformanceTracker, TradingOutcome
from src.domain.backtesting.signal_accuracy_validator import SignalAccuracyValidator, SignalRecord
from src.domain.backtesting.risk_management_simulator import (
    RiskManagementSimulator, Portfolio, Position, StressScenario
)


async def run_complete_backtest():
    """
    Run a complete backtest demonstrating all system capabilities
    """
    print("🚀 GoldenSignalsAI V2 - Complete Backtesting System Demo")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("1. Real market data fetching")
    print("2. Market microstructure simulation")
    print("3. Adaptive agent learning")
    print("4. Signal accuracy validation")
    print("5. Risk management and stress testing")
    print("6. Comprehensive reporting")
    print("\n" + "=" * 70)

    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    initial_capital = 100000
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Phase 1: Initialize Components
    print("\n📦 Phase 1: Initializing Components...")

    data_manager = EnhancedDataManager()
    market_simulator = MarketMicrostructureSimulator({
        'base_spread_bps': 2,
        'market_impact_factor': 0.1,
        'latency_ms': 50
    })
    signal_validator = SignalAccuracyValidator()
    risk_manager = RiskManagementSimulator({
        'var_confidence_levels': [0.95, 0.99],
        'risk_free_rate': 0.02
    })
    agent_tracker = AgentPerformanceTracker()

    print("✅ All components initialized")

    # Phase 2: Fetch Real Market Data
    print("\n📊 Phase 2: Fetching Real Market Data...")

    market_data = {}
    data_quality_scores = {}

    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            data = await data_manager.fetch_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            market_data[symbol] = data

            # Validate data quality
            quality = data_manager.validate_data_quality(data)
            data_quality_scores[symbol] = quality['overall_score']
            print(f"  ✅ {symbol}: {len(data)} days, Quality: {quality['overall_score']:.1f}%")
        except Exception as e:
            print(f"  ❌ Failed to fetch {symbol}: {e}")

    avg_quality = np.mean(list(data_quality_scores.values()))
    print(f"\n  Average Data Quality: {avg_quality:.1f}%")

    # Phase 3: Initialize Portfolio and Agents
    print("\n💼 Phase 3: Setting Up Portfolio and Adaptive Agents...")

    portfolio = Portfolio(
        cash=initial_capital,
        positions={},
        historical_returns=[],
        risk_limits={
            'var_95': -0.05,
            'max_drawdown': -0.10,
            'leverage': 2.0
        }
    )

    # Create multiple adaptive agents
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

    for agent in agents:
        agent_tracker.register_agent(agent)

    print(f"✅ Portfolio initialized with ${initial_capital:,.2f}")
    print(f"✅ {len(agents)} adaptive agents created")

    # Phase 4: Run Backtest Simulation
    print("\n📈 Phase 4: Running Backtest Simulation...")
    print("  Processing trading days with adaptive learning...\n")

    # Get trading days
    first_symbol_data = market_data[symbols[0]]
    trading_days = first_symbol_data.index[50:]  # Skip warmup period

    trades = []
    daily_values = [initial_capital]

    for i, current_date in enumerate(trading_days):
        # Update portfolio prices
        for symbol in symbols:
            if symbol in portfolio.positions and symbol in market_data:
                current_price = market_data[symbol].loc[current_date, 'close']
                portfolio.positions[symbol].current_price = current_price

        # Make trading decisions
        for symbol in symbols:
            if symbol not in market_data:
                continue

            # Get historical data up to current date
            symbol_data = market_data[symbol].loc[:current_date]

            # Get decisions from all agents
            for agent in agents:
                decision = await agent.make_decision(symbol_data, symbol)

                # Execute if not HOLD and passes risk checks
                if decision.action != 'HOLD' and decision.confidence > 0.6:
                    # Create order
                    if decision.action == 'BUY' and portfolio.cash > 10000:
                        order = Order(
                            order_id=f"{agent.agent_id}_{symbol}_{i}",
                            symbol=symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=int(10000 / symbol_data['close'].iloc[-1])
                        )
                    elif decision.action == 'SELL' and symbol in portfolio.positions:
                        order = Order(
                            order_id=f"{agent.agent_id}_{symbol}_{i}",
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=portfolio.positions[symbol].quantity
                        )
                    else:
                        continue

                    # Simulate market microstructure
                    order_book = market_simulator.simulate_order_book(
                        symbol=symbol,
                        mid_price=symbol_data['close'].iloc[-1],
                        volume=symbol_data['volume'].iloc[-1],
                        volatility=symbol_data['close'].pct_change().rolling(20).std().iloc[-1],
                        timestamp=current_date
                    )

                    # Execute order
                    executed_order = await market_simulator.execute_order(
                        order,
                        order_book,
                        {'avg_daily_volume': symbol_data['volume'].mean(), 'volatility': 0.02}
                    )

                    # Update portfolio
                    if executed_order.status.value == 'FILLED':
                        if order.side == OrderSide.BUY:
                            cost = executed_order.filled_quantity * executed_order.avg_fill_price
                            portfolio.cash -= cost
                            portfolio.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=executed_order.filled_quantity,
                                entry_price=executed_order.avg_fill_price,
                                current_price=executed_order.avg_fill_price,
                                entry_time=current_date
                            )
                        else:  # SELL
                            proceeds = executed_order.filled_quantity * executed_order.avg_fill_price
                            portfolio.cash += proceeds
                            del portfolio.positions[symbol]

                        trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': order.side.value,
                            'quantity': executed_order.filled_quantity,
                            'price': executed_order.avg_fill_price,
                            'agent': agent.agent_id
                        })

                        # Record signal for validation
                        signal = SignalRecord(
                            timestamp=current_date,
                            symbol=symbol,
                            signal_type=decision.action,
                            confidence=decision.confidence,
                            predicted_direction='UP' if decision.action == 'BUY' else 'DOWN',
                            predicted_magnitude=abs(decision.predicted_return),
                            source=agent.agent_id
                        )
                        signal_validator.record_signal(signal)

                    # Record outcome for learning (if not last day)
                    if i < len(trading_days) - 1:
                        next_date = trading_days[i + 1]
                        next_price = market_data[symbol].loc[next_date, 'close']
                        actual_return = (next_price - symbol_data['close'].iloc[-1]) / symbol_data['close'].iloc[-1]

                        if decision.action == 'SELL':
                            actual_return = -actual_return

                        outcome = TradingOutcome(
                            decision=decision,
                            actual_return=actual_return,
                            market_conditions={'volatility': 0.02},
                            execution_quality=0.95 if executed_order.status.value == 'FILLED' else 0.5,
                            timestamp=current_date
                        )

                        agent.record_outcome(outcome)
                        agent_tracker.track_decision(agent.agent_id, decision, outcome)

        # Calculate daily portfolio value
        portfolio_value = portfolio.total_value
        daily_values.append(portfolio_value)

        if len(daily_values) > 1:
            daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
            portfolio.historical_returns.append(daily_return)

        # Progress update
        if (i + 1) % 10 == 0:
            current_return = (portfolio_value - initial_capital) / initial_capital * 100
            print(f"  Day {i+1}/{len(trading_days)}: Portfolio ${portfolio_value:,.2f} ({current_return:+.1f}%)")

    # Phase 5: Risk Analysis
    print("\n⚠️ Phase 5: Risk Analysis and Stress Testing...")

    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_risk_metrics(portfolio)

    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
    print(f"  VaR (99%): {risk_metrics.get('var_99', 0):.2%}")
    print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
    print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Current Leverage: {risk_metrics.get('leverage', 0):.2f}x")

    # Run stress tests
    stress_scenarios = [
        StressScenario(
            name="Market Crash",
            description="2008-style crisis",
            market_shock={s: -0.25 for s in symbols},
            volatility_multiplier=3.0,
            correlation_breakdown=True,
            duration_days=5
        ),
        StressScenario(
            name="Tech Bubble Burst",
            description="2000-style tech selloff",
            market_shock={s: -0.35 for s in symbols},
            volatility_multiplier=2.5,
            correlation_breakdown=True,
            duration_days=30
        )
    ]

    print("\nStress Test Results:")
    for scenario in stress_scenarios:
        result = risk_manager.run_stress_test(portfolio, scenario, market_data)
        print(f"  {scenario.name}: {result['loss_percentage']:.1f}% potential loss")

    # Check circuit breakers
    triggered = risk_manager.check_circuit_breakers(portfolio, market_data, datetime.now())
    if triggered:
        print("\n⚠️  Circuit Breakers Triggered:")
        for breaker in triggered:
            print(f"  - {breaker['breaker']}: {breaker['action']}")
    else:
        print("\n✅ No circuit breakers triggered")

    # Phase 6: Performance Summary
    print("\n📊 Phase 6: Final Performance Summary")
    print("=" * 70)

    # Portfolio Performance
    total_return = (portfolio.total_value - initial_capital) / initial_capital
    print(f"\nPortfolio Performance:")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Value: ${portfolio.total_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Total Trades: {len(trades)}")

    # Agent Performance
    print(f"\nAgent Performance:")
    rankings = agent_tracker.get_agent_ranking()
    for rank, (agent_id, score) in enumerate(rankings, 1):
        agent = agent_tracker.agents[agent_id]
        metrics = agent.calculate_performance_metrics()
        print(f"  {rank}. {agent_id}:")
        print(f"     Score: {score:.3f}")
        print(f"     Win Rate: {metrics.win_rate:.1%}")
        print(f"     Model Version: {agent.model_version}")

    # Signal Accuracy
    if signal_validator.signals:
        accuracy_report = signal_validator.calculate_accuracy_metrics()
        print(f"\nSignal Accuracy:")
        print(f"  Direction Accuracy: {accuracy_report['direction_accuracy']:.1%}")
        print(f"  Win Rate: {accuracy_report['win_rate']:.1%}")
        print(f"  Avg Profit per Signal: {accuracy_report['avg_profit_per_signal']:.2%}")

    # Execution Quality
    exec_analytics = market_simulator.get_execution_analytics()
    if exec_analytics:
        print(f"\nExecution Quality:")
        print(f"  Fill Rate: {exec_analytics.get('fill_rate', 0):.1%}")
        print(f"  Avg Slippage: {exec_analytics.get('avg_slippage_bps', 0):.1f} bps")
        print(f"  Avg Execution Time: {exec_analytics.get('avg_execution_time_ms', 0):.1f} ms")

    # Save Results
    results = {
        'summary': {
            'initial_capital': initial_capital,
            'final_value': portfolio.total_value,
            'total_return': total_return,
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
            'max_drawdown': risk_metrics.get('max_drawdown', 0),
            'total_trades': len(trades)
        },
        'trades': trades,
        'daily_values': daily_values,
        'risk_metrics': risk_metrics,
        'agent_performance': {
            agent_id: {
                'score': score,
                'metrics': agent_tracker.agents[agent_id].calculate_performance_metrics().__dict__
            }
            for agent_id, score in rankings
        }
    }

    with open('complete_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ Results saved to complete_backtest_results.json")

    # Plot Results
    try:
        plt.figure(figsize=(12, 6))

        # Portfolio value over time
        plt.subplot(1, 2, 1)
        plt.plot(daily_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)

        # Daily returns distribution
        plt.subplot(1, 2, 2)
        if portfolio.historical_returns:
            plt.hist(portfolio.historical_returns, bins=30, alpha=0.7)
            plt.title('Daily Returns Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        print("📈 Performance charts saved to backtest_results.png")
    except Exception as e:
        print(f"Could not generate charts: {e}")

    print("\n🎉 Complete Backtest Demo Finished Successfully!")
    print("\nKey Takeaways:")
    print("1. Real market data was fetched and validated")
    print("2. Market microstructure was realistically simulated")
    print("3. Agents learned and adapted during the backtest")
    print("4. Risk was monitored and stress tested")
    print("5. Comprehensive results were generated")

    return results


if __name__ == "__main__":
    # Run the complete backtest
    asyncio.run(run_complete_backtest())
