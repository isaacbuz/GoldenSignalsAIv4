"""
Agent Factory for GoldenSignalsAI
Centralized agent creation and management
"""

from typing import Dict, Any, Optional, Type, List
import importlib
import logging
from dataclasses import dataclass
from enum import Enum

from agents.core.unified_base_agent import UnifiedBaseAgent, AgentType


@dataclass
class AgentSpec:
    """Specification for creating an agent"""
    agent_class: str  # Full module path to agent class
    agent_type: AgentType
    default_config: Dict[str, Any]
    capabilities: List[str]
    dependencies: List[str] = None  # Other agents this depends on


class AgentFactory:
    """Factory for creating and managing agents"""
    
    # Agent registry mapping agent names to specifications
    AGENT_REGISTRY = {
        # Technical Analysis Agents
        'momentum_agent': AgentSpec(
            agent_class='src.agents.technical.momentum_agent.MomentumAgent',
            agent_type=AgentType.TECHNICAL,
            default_config={
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'momentum_threshold': 0.02
            },
            capabilities=['momentum_analysis', 'trend_following', 'breakout_detection']
        ),
        
        'mean_reversion_agent': AgentSpec(
            agent_class='src.agents.technical.mean_reversion_agent.MeanReversionAgent',
            agent_type=AgentType.TECHNICAL,
            default_config={
                'bb_period': 20,
                'bb_std': 2.0,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            },
            capabilities=['mean_reversion', 'oversold_detection', 'overbought_detection']
        ),
        
        'pattern_recognition_agent': AgentSpec(
            agent_class='src.agents.technical.pattern_agent.PatternRecognitionAgent',
            agent_type=AgentType.TECHNICAL,
            default_config={
                'min_pattern_strength': 0.7,
                'lookback_periods': [20, 50, 100],
                'pattern_types': ['head_shoulders', 'triangles', 'flags', 'channels']
            },
            capabilities=['pattern_recognition', 'chart_analysis', 'support_resistance']
        ),
        
        # Sentiment Analysis Agents
        'news_sentiment_agent': AgentSpec(
            agent_class='src.agents.sentiment.news_agent.NewsSentimentAgent',
            agent_type=AgentType.SENTIMENT,
            default_config={
                'news_sources': ['newsapi', 'finnhub', 'benzinga'],
                'sentiment_model': 'finbert',
                'lookback_hours': 24,
                'min_relevance_score': 0.6
            },
            capabilities=['news_sentiment', 'headline_analysis', 'event_detection']
        ),
        
        'social_sentiment_agent': AgentSpec(
            agent_class='src.agents.sentiment.social_agent.SocialSentimentAgent',
            agent_type=AgentType.SENTIMENT,
            default_config={
                'platforms': ['reddit', 'twitter', 'stocktwits'],
                'subreddits': ['wallstreetbets', 'stocks', 'investing'],
                'sentiment_threshold': 0.6,
                'volume_spike_threshold': 2.0
            },
            capabilities=['social_sentiment', 'crowd_psychology', 'viral_detection']
        ),
        
        # ML/AI Agents
        'ml_predictor_agent': AgentSpec(
            agent_class='src.agents.ml.predictor_agent.MLPredictorAgent',
            agent_type=AgentType.ML,
            default_config={
                'models': ['lstm', 'xgboost', 'random_forest'],
                'prediction_horizons': ['5m', '15m', '1h', '1d'],
                'feature_engineering': True,
                'ensemble_method': 'weighted_average'
            },
            capabilities=['price_prediction', 'volatility_forecast', 'trend_prediction']
        ),
        
        'reinforcement_learning_agent': AgentSpec(
            agent_class='src.agents.ml.rl_agent.ReinforcementLearningAgent',
            agent_type=AgentType.ML,
            default_config={
                'algorithm': 'PPO',
                'state_features': 50,
                'action_space': ['buy', 'sell', 'hold'],
                'reward_function': 'sharpe_ratio',
                'training_episodes': 1000
            },
            capabilities=['rl_trading', 'adaptive_strategy', 'online_learning']
        ),
        
        # Options Analysis Agents
        'options_flow_agent': AgentSpec(
            agent_class='src.agents.options.flow_agent.OptionsFlowAgent',
            agent_type=AgentType.OPTIONS,
            default_config={
                'min_premium': 100000,
                'unusual_activity_threshold': 3.0,
                'track_sweeps': True,
                'track_blocks': True
            },
            capabilities=['options_flow', 'unusual_activity', 'smart_money_tracking']
        ),
        
        'greeks_agent': AgentSpec(
            agent_class='src.agents.options.greeks_agent.GreeksAgent',
            agent_type=AgentType.OPTIONS,
            default_config={
                'calculate_iv': True,
                'gamma_threshold': 0.05,
                'vega_threshold': 0.1,
                'track_skew': True
            },
            capabilities=['greeks_analysis', 'iv_analysis', 'options_pricing']
        ),
        
        # Risk Management Agents
        'portfolio_risk_agent': AgentSpec(
            agent_class='src.agents.risk.portfolio_agent.PortfolioRiskAgent',
            agent_type=AgentType.RISK,
            default_config={
                'var_confidence': 0.95,
                'max_position_size': 0.1,
                'max_sector_exposure': 0.3,
                'correlation_threshold': 0.7
            },
            capabilities=['portfolio_risk', 'position_sizing', 'risk_metrics'],
            dependencies=['market_data_agent']
        ),
        
        'stop_loss_agent': AgentSpec(
            agent_class='src.agents.risk.stop_loss_agent.StopLossAgent',
            agent_type=AgentType.RISK,
            default_config={
                'atr_multiplier': 2.0,
                'trailing_stop_activation': 0.02,
                'max_loss_per_trade': 0.02,
                'use_options_hedge': True
            },
            capabilities=['stop_loss', 'trailing_stops', 'hedging_strategies']
        ),
        
        # Market Analysis Agents
        'market_regime_agent': AgentSpec(
            agent_class='src.agents.market.regime_agent.MarketRegimeAgent',
            agent_type=AgentType.MARKET,
            default_config={
                'regime_lookback': 60,
                'volatility_bands': [0.1, 0.2, 0.3],
                'trend_strength_threshold': 0.3,
                'regime_change_sensitivity': 0.8
            },
            capabilities=['regime_detection', 'market_classification', 'volatility_analysis']
        ),
        
        'correlation_agent': AgentSpec(
            agent_class='src.agents.market.correlation_agent.CorrelationAgent',
            agent_type=AgentType.MARKET,
            default_config={
                'correlation_window': 30,
                'min_correlation': 0.5,
                'track_sectors': True,
                'track_indices': True
            },
            capabilities=['correlation_analysis', 'sector_rotation', 'pairs_identification']
        ),
        
        # Orchestration Agents
        'agent_orchestrator': AgentSpec(
            agent_class='src.agents.orchestration.agent_orchestrator.AgentOrchestrator',
            agent_type=AgentType.ORCHESTRATOR,
            default_config={
                'max_parallel_workflows': 10,
                'default_timeout': 60.0,
                'retry_policy': {'max_retries': 3, 'backoff': 'exponential'},
                'load_balancing': 'round_robin'
            },
            capabilities=['workflow_orchestration', 'agent_coordination', 'task_scheduling']
        ),
        
        'meta_orchestrator': AgentSpec(
            agent_class='src.agents.orchestration.meta_orchestrator.MetaOrchestrator',
            agent_type=AgentType.ORCHESTRATOR,
            default_config={
                'adaptation_rate': 0.1,
                'strategy_evaluation_period': 100,
                'max_strategies_per_regime': 5,
                'performance_tracking': True
            },
            capabilities=['meta_orchestration', 'strategy_selection', 'regime_adaptation']
        ),
        
        # Specialized Agents
        'arbitrage_agent': AgentSpec(
            agent_class='src.agents.specialized.arbitrage_agent.ArbitrageAgent',
            agent_type=AgentType.SPECIALIZED,
            default_config={
                'min_spread': 0.001,
                'execution_speed': 'fast',
                'track_triangular': True,
                'track_statistical': True
            },
            capabilities=['arbitrage_detection', 'spread_analysis', 'cross_market_opportunities']
        ),
        
        'event_driven_agent': AgentSpec(
            agent_class='src.agents.specialized.event_agent.EventDrivenAgent',
            agent_type=AgentType.SPECIALIZED,
            default_config={
                'event_types': ['earnings', 'economic_data', 'corporate_actions'],
                'pre_event_window': 5,
                'post_event_window': 5,
                'min_expected_move': 0.02
            },
            capabilities=['event_trading', 'catalyst_detection', 'earnings_plays']
        ),
        
        # Crypto-specific Agents
        'defi_agent': AgentSpec(
            agent_class='src.agents.crypto.defi_agent.DeFiAgent',
            agent_type=AgentType.SPECIALIZED,
            default_config={
                'protocols': ['uniswap', 'aave', 'compound'],
                'min_yield': 0.05,
                'gas_optimization': True,
                'impermanent_loss_threshold': 0.02
            },
            capabilities=['defi_opportunities', 'yield_farming', 'liquidity_provision']
        ),
        
        'onchain_agent': AgentSpec(
            agent_class='src.agents.crypto.onchain_agent.OnChainAgent',
            agent_type=AgentType.SPECIALIZED,
            default_config={
                'track_whale_movements': True,
                'min_transaction_size': 100000,
                'track_smart_contracts': True,
                'mempool_monitoring': True
            },
            capabilities=['onchain_analysis', 'whale_tracking', 'smart_money_flow']
        )
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the agent factory"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.created_agents: Dict[str, UnifiedBaseAgent] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
        
    def create_agent(self, 
                    agent_name: str,
                    agent_id: Optional[str] = None,
                    custom_config: Optional[Dict[str, Any]] = None) -> UnifiedBaseAgent:
        """Create an agent instance"""
        if agent_name not in self.AGENT_REGISTRY:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        spec = self.AGENT_REGISTRY[agent_name]
        
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"{agent_name}_{id(spec)}"
        
        # Check if agent already exists
        if agent_id in self.created_agents:
            self.logger.info(f"Returning existing agent: {agent_id}")
            return self.created_agents[agent_id]
        
        # Merge configurations
        config = spec.default_config.copy()
        if custom_config:
            config.update(custom_config)
        
        # Add global config
        config.update(self.config)
        
        try:
            # Dynamically import the agent class
            module_path, class_name = spec.agent_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            
            # Create the agent instance
            agent = agent_class(agent_id=agent_id, config=config)
            
            # Store the agent
            self.created_agents[agent_id] = agent
            
            # Track dependencies
            if spec.dependencies:
                self.agent_dependencies[agent_id] = spec.dependencies
            
            self.logger.info(f"Created agent: {agent_id} ({agent_name})")
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_name}: {e}")
            # Return a mock agent for now
            from agents.core.unified_base_agent import UnifiedBaseAgent
            return UnifiedBaseAgent(agent_id=agent_id, agent_type=spec.agent_type, config=config)
    
    def create_agent_ensemble(self,
                            ensemble_name: str,
                            agent_names: List[str],
                            orchestrator_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an ensemble of agents with an orchestrator"""
        ensemble = {
            'name': ensemble_name,
            'agents': {},
            'orchestrator': None
        }
        
        # Create individual agents
        for agent_name in agent_names:
            agent_id = f"{ensemble_name}_{agent_name}"
            agent = self.create_agent(agent_name, agent_id)
            ensemble['agents'][agent_id] = agent
        
        # Create orchestrator
        orch_config = orchestrator_config or {}
        orch_config['managed_agents'] = list(ensemble['agents'].keys())
        
        orchestrator = self.create_agent(
            'agent_orchestrator',
            f"{ensemble_name}_orchestrator",
            orch_config
        )
        ensemble['orchestrator'] = orchestrator
        
        self.logger.info(f"Created ensemble: {ensemble_name} with {len(agent_names)} agents")
        
        return ensemble
    
    def create_strategy_agents(self, strategy: str) -> Dict[str, UnifiedBaseAgent]:
        """Create all agents needed for a specific strategy"""
        strategy_agents = {
            'momentum': ['momentum_agent', 'market_regime_agent', 'portfolio_risk_agent'],
            'mean_reversion': ['mean_reversion_agent', 'correlation_agent', 'stop_loss_agent'],
            'sentiment_driven': ['news_sentiment_agent', 'social_sentiment_agent', 'ml_predictor_agent'],
            'options_flow': ['options_flow_agent', 'greeks_agent', 'portfolio_risk_agent'],
            'multi_strategy': [
                'momentum_agent', 'mean_reversion_agent', 'sentiment_agent',
                'ml_predictor_agent', 'portfolio_risk_agent', 'meta_orchestrator'
            ]
        }
        
        agent_names = strategy_agents.get(strategy, [])
        if not agent_names:
            self.logger.warning(f"No predefined agents for strategy: {strategy}")
            return {}
        
        agents = {}
        for agent_name in agent_names:
            agent = self.create_agent(agent_name, f"{strategy}_{agent_name}")
            agents[agent.agent_id] = agent
        
        return agents
    
    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """Get capabilities of an agent"""
        if agent_name not in self.AGENT_REGISTRY:
            return []
        
        return self.AGENT_REGISTRY[agent_name].capabilities
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Find all agents with a specific capability"""
        agents = []
        
        for agent_name, spec in self.AGENT_REGISTRY.items():
            if capability in spec.capabilities:
                agents.append(agent_name)
        
        return agents
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[str]:
        """Get all agents of a specific type"""
        agents = []
        
        for agent_name, spec in self.AGENT_REGISTRY.items():
            if spec.agent_type == agent_type:
                agents.append(agent_name)
        
        return agents
    
    def shutdown_all_agents(self):
        """Shutdown all created agents"""
        for agent_id, agent in self.created_agents.items():
            try:
                agent.shutdown()
                self.logger.info(f"Shutdown agent: {agent_id}")
            except Exception as e:
                self.logger.error(f"Error shutting down agent {agent_id}: {e}")
        
        self.created_agents.clear()
        self.agent_dependencies.clear()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            'total_agents': len(self.created_agents),
            'agents_by_type': {},
            'agent_details': []
        }
        
        # Count by type
        for agent_id, agent in self.created_agents.items():
            agent_type = str(agent.agent_type.value)
            if agent_type not in status['agents_by_type']:
                status['agents_by_type'][agent_type] = 0
            status['agents_by_type'][agent_type] += 1
            
            # Add details
            status['agent_details'].append({
                'agent_id': agent_id,
                'agent_type': agent_type,
                'capabilities': agent.capabilities,
                'status': 'active' if hasattr(agent, 'is_active') and agent.is_active else 'unknown'
            })
        
        return status
    
    @classmethod
    def get_available_agents(cls) -> List[str]:
        """Get list of all available agent types"""
        return list(cls.AGENT_REGISTRY.keys())
    
    @classmethod
    def get_agent_info(cls, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent type"""
        if agent_name not in cls.AGENT_REGISTRY:
            return None
        
        spec = cls.AGENT_REGISTRY[agent_name]
        return {
            'name': agent_name,
            'type': spec.agent_type.value,
            'capabilities': spec.capabilities,
            'default_config': spec.default_config,
            'dependencies': spec.dependencies or []
        }


# Singleton instance
_factory_instance: Optional[AgentFactory] = None


def get_agent_factory(config: Dict[str, Any] = None) -> AgentFactory:
    """Get or create the singleton agent factory"""
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = AgentFactory(config)
    
    return _factory_instance 