"""
Demo of Comprehensive Agent System for GoldenSignalsAI
Shows how all agents work together
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List

# Import agent components
from src.agents.core.agent_factory import get_agent_factory
from src.agents.orchestration.agent_orchestrator import (
    AgentOrchestrator, OrchestrationStrategy, WorkflowDefinition, AgentTask
)
from src.agents.orchestration.meta_orchestrator import (
    MetaOrchestrator, MetaStrategy, MetaWorkflow
)
from src.agents.core.unified_base_agent import MessagePriority


def generate_sample_market_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate sample market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days * 78, freq='5min')  # 78 5-min bars per day
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.002, len(dates))
    price = 100 * np.exp(np.cumsum(returns))
    
    # Add some trends
    trend = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 5
    price = price + trend
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': price * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'close': price,
        'volume': np.random.lognormal(10, 1, len(dates))
    })
    
    data.set_index('timestamp', inplace=True)
    return data


async def demo_individual_agents():
    """Demo individual agent capabilities"""
    print("\n=== INDIVIDUAL AGENT DEMO ===\n")
    
    # Create agent factory
    factory = get_agent_factory()
    
    # Create a momentum agent
    momentum_agent = factory.create_agent('momentum_agent', 'demo_momentum_001')
    
    # Generate sample data
    data = generate_sample_market_data('AAPL')
    
    # Test momentum analysis
    print("1. Testing Momentum Analysis:")
    result = await momentum_agent.process_request({
        'type': 'analyze',
        'symbol': 'AAPL',
        'data': data
    })
    
    print(f"   Momentum Strength: {result.get('strength', 0):.3f}")
    print(f"   Interpretation: {result.get('interpretation', 'N/A')}")
    print(f"   Key Indicators:")
    indicators = result.get('indicators', {})
    for key in ['rsi', 'macd', 'adx', 'volume_ratio']:
        if key in indicators:
            print(f"     - {key}: {indicators[key]:.2f}")
    
    # Generate signals
    print("\n2. Testing Signal Generation:")
    signal_result = await momentum_agent.process_request({
        'type': 'generate_signals',
        'symbol': 'AAPL',
        'data': data
    })
    
    signals = signal_result.get('signals', [])
    print(f"   Generated {len(signals)} signals")
    for signal in signals[:3]:  # Show first 3
        print(f"   - {signal['type']}: {signal['action']} @ ${signal['price']:.2f} (confidence: {signal['confidence']:.2f})")


async def demo_agent_orchestration():
    """Demo agent orchestration"""
    print("\n\n=== AGENT ORCHESTRATION DEMO ===\n")
    
    factory = get_agent_factory()
    
    # Create orchestrator
    orchestrator = factory.create_agent('agent_orchestrator', 'demo_orchestrator_001')
    
    # Create multiple agents
    agents = {
        'momentum': factory.create_agent('momentum_agent', 'orch_momentum_001'),
        'mean_reversion': factory.create_agent('mean_reversion_agent', 'orch_mean_rev_001'),
        'risk': factory.create_agent('portfolio_risk_agent', 'orch_risk_001')
    }
    
    # Register agents with orchestrator
    for agent_id, agent in agents.items():
        await orchestrator.handle_agent_registration({
            'payload': {
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type.value,
                'capabilities': list(agent.capabilities.keys())
            }
        })
    
    # Create a workflow
    data = generate_sample_market_data('SPY')
    
    workflow = WorkflowDefinition(
        workflow_id='demo_workflow_001',
        name='Multi-Strategy Analysis',
        tasks=[
            AgentTask(
                task_id='momentum_analysis',
                agent_id='orch_momentum_001',
                task_type='analyze_momentum',
                params={'symbol': 'SPY', 'data': data},
                priority=MessagePriority.HIGH
            ),
            AgentTask(
                task_id='mean_rev_analysis',
                agent_id='orch_mean_rev_001',
                task_type='analyze_mean_reversion',
                params={'symbol': 'SPY', 'data': data},
                priority=MessagePriority.HIGH
            ),
            AgentTask(
                task_id='risk_assessment',
                agent_id='orch_risk_001',
                task_type='assess_risk',
                params={'symbol': 'SPY', 'data': data},
                dependencies=['momentum_analysis', 'mean_rev_analysis'],
                priority=MessagePriority.NORMAL
            )
        ],
        strategy=OrchestrationStrategy.HIERARCHICAL,
        timeout=60.0
    )
    
    print("Executing hierarchical workflow...")
    result = await orchestrator.handle_execute_workflow({
        'payload': {'workflow': workflow.__dict__}
    })
    
    print(f"Workflow completed: {result.get('completed', False)}")
    print(f"Execution order: {result.get('execution_order', [])}")


async def demo_meta_orchestration():
    """Demo meta-level orchestration"""
    print("\n\n=== META ORCHESTRATION DEMO ===\n")
    
    factory = get_agent_factory()
    
    # Create meta orchestrator
    meta_orchestrator = factory.create_agent('meta_orchestrator', 'demo_meta_001')
    
    # Create market context
    market_context = {
        'symbols': ['SPY', 'QQQ', 'IWM'],
        'timeframe': '5m',
        'volatility': 0.25,
        'trend_strength': 0.6,
        'volume_ratio': 1.3,
        'sentiment_score': 0.4
    }
    
    # Create meta workflow
    meta_workflow = MetaWorkflow(
        workflow_id='meta_demo_001',
        name='Adaptive Market Strategy',
        meta_strategy=MetaStrategy.MARKET_ADAPTIVE,
        sub_workflows=[],  # Will be created by meta orchestrator
        market_context=market_context,
        risk_parameters={
            'max_portfolio_risk': 0.02,
            'risk_per_trade': 0.01,
            'max_strategy_drawdown': 0.05
        },
        performance_targets={
            'min_return': 0.001,
            'target_return': 0.02,
            'max_risk': 0.03
        },
        coordination_rules=[
            {
                'group_name': 'correlated_strategies',
                'strategies': ['momentum', 'trend_following'],
                'share_results': True
            }
        ]
    )
    
    print("Executing meta workflow with adaptive strategy...")
    
    # Detect market regime
    regime_result = await meta_orchestrator.handle_update_market_regime({
        'payload': {'market_data': market_context}
    })
    
    print(f"Detected Market Regime: {regime_result['regime']['regime_type']}")
    print(f"Regime Confidence: {regime_result['regime']['confidence']:.2f}")
    
    # Execute meta workflow
    meta_result = await meta_orchestrator.handle_execute_meta_workflow({
        'payload': {'meta_workflow': meta_workflow.__dict__}
    })
    
    print(f"\nSelected Strategies: {meta_result.get('strategies_used', [])}")
    print(f"Performance Metrics:")
    metrics = meta_result.get('performance_metrics', {})
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.3f}")


async def demo_agent_ensemble():
    """Demo agent ensemble for complex decision making"""
    print("\n\n=== AGENT ENSEMBLE DEMO ===\n")
    
    factory = get_agent_factory()
    
    # Create an ensemble for sentiment-driven trading
    ensemble = factory.create_agent_ensemble(
        ensemble_name='sentiment_trading',
        agent_names=['news_sentiment_agent', 'social_sentiment_agent', 'ml_predictor_agent'],
        orchestrator_config={
            'coordination_strategy': 'ensemble',
            'voting_method': 'weighted_confidence'
        }
    )
    
    print(f"Created ensemble with {len(ensemble['agents'])} agents")
    print(f"Orchestrator: {ensemble['orchestrator'].agent_id}")
    
    # Show agent capabilities
    print("\nEnsemble Capabilities:")
    for agent_id, agent in ensemble['agents'].items():
        print(f"  {agent_id}: {list(agent.capabilities.keys())[:3]}...")


async def demo_complete_system():
    """Demo complete system integration"""
    print("\n\n=== COMPLETE SYSTEM INTEGRATION DEMO ===\n")
    
    factory = get_agent_factory()
    
    # Get available agents
    available = factory.get_available_agents()
    print(f"Total available agent types: {len(available)}")
    
    # Show agents by type
    from src.agents.core.unified_base_agent import AgentType
    for agent_type in AgentType:
        agents = factory.get_agents_by_type(agent_type)
        print(f"  {agent_type.value}: {len(agents)} agents")
    
    # Find agents with specific capabilities
    print("\nAgents with momentum analysis capability:")
    momentum_agents = factory.get_agents_by_capability('momentum_analysis')
    for agent in momentum_agents:
        info = factory.get_agent_info(agent)
        print(f"  - {agent}: {info['type']}")
    
    # Create multi-strategy setup
    strategies = ['momentum', 'mean_reversion', 'sentiment_driven']
    all_agents = {}
    
    print("\nCreating agents for multiple strategies:")
    for strategy in strategies:
        agents = factory.create_strategy_agents(strategy)
        all_agents.update(agents)
        print(f"  {strategy}: {len(agents)} agents created")
    
    # Show system status
    status = factory.get_agent_status()
    print(f"\nSystem Status:")
    print(f"  Total active agents: {status['total_agents']}")
    print(f"  Agents by type: {status['agents_by_type']}")


def print_demo_summary():
    """Print summary of the demo"""
    print("\n\n" + "="*60)
    print("COMPREHENSIVE AGENT SYSTEM DEMO COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("1. Individual Agent Capabilities")
    print("   - Momentum analysis and signal generation")
    print("   - Technical indicator calculation")
    print("   - Risk assessment")
    print("\n2. Agent Orchestration")
    print("   - Hierarchical workflow execution")
    print("   - Task dependencies and coordination")
    print("   - Parallel and sequential strategies")
    print("\n3. Meta-Level Orchestration")
    print("   - Market regime detection")
    print("   - Adaptive strategy selection")
    print("   - Performance optimization")
    print("\n4. Agent Ensembles")
    print("   - Multi-agent collaboration")
    print("   - Weighted voting mechanisms")
    print("   - Specialized agent groups")
    print("\n5. Complete System Integration")
    print("   - Agent factory management")
    print("   - Capability-based agent discovery")
    print("   - Multi-strategy coordination")
    print("\nThe system is ready for production use with:")
    print("- 20+ specialized agent types")
    print("- Advanced orchestration strategies")
    print("- Real-time adaptation to market conditions")
    print("- Comprehensive risk management")
    print("- Scalable architecture with Ray support")


async def main():
    """Run all demos"""
    print("GOLDEN SIGNALS AI - COMPREHENSIVE AGENT SYSTEM DEMO")
    print("="*60)
    
    try:
        # Run individual demos
        await demo_individual_agents()
        await demo_agent_orchestration()
        await demo_meta_orchestration()
        await demo_agent_ensemble()
        await demo_complete_system()
        
        # Print summary
        print_demo_summary()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 