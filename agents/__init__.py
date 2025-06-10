from typing import Dict, Any

# Experimental and research agents
from .research_agents import BacktestResearchAgent

# Grok-powered agent utilities
from .grok_agents import GrokBacktestCritic, GrokSentimentAgent, GrokStrategyAgent

# Agent-centric monitoring utilities
from .monitoring_agents import AgentPerformanceTracker, AIMonitor

# Strategy tuning utilities
from .strategy_utils import StrategyTuner

# Workflow orchestration utilities
from .workflow_agents import daily_trading_cycle

# Integration utilities (external model adapters, cache)
from .integration_utils import ExternalCache, ModelProviderAdapter, AnthropicClaudeAdapter, MetaLlamaAdapter

# ML model registry utilities
from .ml_registry import ModelRegistry

# Options backtesting utilities
from .options_backtesting import OptionsBacktester

"""
GoldenSignalsAI Trading Agents Package

This package provides a collection of agents for market analysis, trading, and signal generation:

- Data Source Agents: For fetching market data from various providers
- Forecasting Agents: For predicting market movements
- Meta Agents: For combining signals from multiple agents
- Arbitrage Agents: For identifying and executing arbitrage opportunities
- Research Agents: For strategy optimization and backtesting
- Grok-powered Agents: For AI-driven trading research and sentiment analysis
- Monitoring Agents: For tracking agent performance and system health
- Options Trading Agents: For options-specific strategies and backtesting
"""

# Common components
from .common.base.base_agent import BaseAgent
from .common.registry.agent_registry import AgentRegistry, registry
from .common.utils.logging import setup_agent_logger
from .common.utils.validation import validate_market_data, validate_time_series
from .common.utils.persistence import save_model, load_model

# Core Trading Components
# Technical Analysis
from .core.technical import (
    RSIAgent,
    MACDAgent,
    RSIMACDAgent,
    MomentumDivergenceAgent,
    CryptoSignalAgent
)

# New Comprehensive Technical Analysis Agents
try:
    from .core.technical.pattern_agent import PatternAgent
    from .core.technical.breakout_agent import BreakoutAgent
    from .core.technical.mean_reversion_agent import MeanReversionAgent
except ImportError:
    PatternAgent = None
    BreakoutAgent = None
    MeanReversionAgent = None

# Volume Analysis Agents
try:
    from .core.volume.volume_spike_agent import VolumeSpikeAgent
except ImportError:
    VolumeSpikeAgent = None

# Options/Volatility Agents
try:
    from .core.options.gamma_exposure_agent import GammaExposureAgent
    from .core.options.skew_agent import SkewAgent
    from .core.options.iv_rank_agent import IVRankAgent
except ImportError:
    GammaExposureAgent = None
    SkewAgent = None
    IVRankAgent = None

# Macro Regime Agents
try:
    from .core.macro.regime_agent import RegimeAgent
except ImportError:
    RegimeAgent = None

# Flow/Arbitrage Agents
try:
    from .core.flow.etf_arb_agent import ETFArbAgent
    from .core.flow.sector_rotation_agent import SectorRotationAgent
    from .core.flow.whale_trade_agent import WhaleTradeAgent
except ImportError:
    ETFArbAgent = None
    SectorRotationAgent = None
    WhaleTradeAgent = None

# Meta/ML Consensus Agents
try:
    from .meta.meta_consensus_agent import MetaConsensusAgent
    from .meta.meta_signal_agent import MetaSignalAgent
    from .meta.meta_signal_orchestrator import MetaSignalOrchestrator
except ImportError:
    MetaConsensusAgent = None
    MetaSignalAgent = None
    MetaSignalOrchestrator = None

# Sentiment and News Analysis Agents
try:
    from .core.sentiment.news_agent import NewsAgent
except ImportError:
    NewsAgent = None

# Fundamental Analysis
from .core.fundamental import (
    ValuationAgent,
    FinancialMetricsAgent,
    EarningsAnalysisAgent,
    IndustryAnalysisAgent
)

# Sentiment Analysis
from .core.sentiment import (
    NewsAnalysisAgent,
    SocialMediaAgent,
    MarketSentimentAgent,
    AnalystSentimentAgent
)

# Risk Management
from .core.risk import (
    PositionSizingAgent,
    StopLossAgent,
    RiskRewardAgent,
    VolatilityAgent,
    PortfolioRiskAgent
)

# Market Timing
from .core.timing import (
    MarketRegimeAgent,
    LiquidityAnalysisAgent,
    ExecutionOptimizationAgent,
    SignalTimingAgent
)

# Portfolio Management
from .core.portfolio import (
    AllocationAgent,
    RebalancingAgent,
    DiversificationAgent,
    PerformanceTrackingAgent
)

# Infrastructure Components
from .infrastructure.data_sources import (
    DataSourceAgent,
    AlphaVantageAgent,
    FinnhubAgent,
    PolygonAgent,
    BenzingaNewsAgent,
    StockTwitsAgent,
    BloombergAgent,
    DataAggregator
)

from .infrastructure.integration import (
    DataSourceAdapter,
    BrokerAdapter,
    APIIntegrationAgent,
    ExternalCache,
    ModelProviderAdapter,
    AnthropicClaudeAdapter,
    MetaLlamaAdapter
)

from .infrastructure.monitoring import (
    PerformanceMonitorAgent,
    RiskMonitorAgent,
    ComplianceAgent,
    AgentPerformanceTracker,
    AIMonitor
)

# Research Components
from .research.backtesting import (
    BacktestResearchAgent,
    OptionsBacktester
)

from .research.ml import (
    EnhancedSignalAgent,
    PredictiveAnalysisAgent,
    DeepLearningAgent
)

from .research.optimization import (
    StrategyTuner,
    ModelRegistry
)

# Experimental Components
from .experimental.grok import (
    GrokBacktestCritic,
    GrokSentimentAgent,
    GrokStrategyAgent
)

# Version
__version__ = "1.0.0"

__all__ = [
    # Common Components
    'BaseAgent',
    'AgentRegistry',
    'registry',
    'setup_agent_logger',
    'validate_market_data',
    'validate_time_series',
    'save_model',
    'load_model',
    
    # Core Technical Analysis
    'RSIAgent',
    'MACDAgent',
    'RSIMACDAgent',
    'MomentumDivergenceAgent',
    'CryptoSignalAgent',
    
    # New Comprehensive Technical Analysis
    'PatternAgent',
    'BreakoutAgent', 
    'MeanReversionAgent',
    
    # Volume Analysis
    'VolumeSpikeAgent',
    
    # Options/Volatility Analysis
    'GammaExposureAgent',
    'SkewAgent',
    'IVRankAgent',
    
    # Macro Regime Analysis
    'RegimeAgent',
    
    # Flow/Arbitrage Analysis
    'ETFArbAgent',
    'SectorRotationAgent',
    'WhaleTradeAgent',
    
    # Meta/ML Consensus
    'MetaConsensusAgent',
    'MetaSignalAgent',
    'MetaSignalOrchestrator',
    
    # News and Sentiment Analysis  
    'NewsAgent',
    
    # Core Fundamental Analysis
    'ValuationAgent',
    'FinancialMetricsAgent',
    'EarningsAnalysisAgent',
    'IndustryAnalysisAgent',
    
    # Core Sentiment Analysis
    'NewsAnalysisAgent',
    'SocialMediaAgent',
    'MarketSentimentAgent',
    'AnalystSentimentAgent',
    
    # Core Risk Management
    'PositionSizingAgent',
    'StopLossAgent',
    'RiskRewardAgent',
    'VolatilityAgent',
    'PortfolioRiskAgent',
    
    # Core Market Timing
    'MarketRegimeAgent',
    'LiquidityAnalysisAgent',
    'ExecutionOptimizationAgent',
    'SignalTimingAgent',
    
    # Core Portfolio Management
    'AllocationAgent',
    'RebalancingAgent',
    'DiversificationAgent',
    'PerformanceTrackingAgent',
    
    # Infrastructure - Data Sources
    'DataSourceAgent',
    'AlphaVantageAgent',
    'FinnhubAgent',
    'PolygonAgent',
    'BenzingaNewsAgent',
    'StockTwitsAgent',
    'BloombergAgent',
    'DataAggregator',
    
    # Infrastructure - Integration
    'DataSourceAdapter',
    'BrokerAdapter',
    'APIIntegrationAgent',
    'ExternalCache',
    'ModelProviderAdapter',
    'AnthropicClaudeAdapter',
    'MetaLlamaAdapter',
    
    # Infrastructure - Monitoring
    'PerformanceMonitorAgent',
    'RiskMonitorAgent',
    'ComplianceAgent',
    'AgentPerformanceTracker',
    'AIMonitor',
    
    # Research Components
    'BacktestResearchAgent',
    'OptionsBacktester',
    'EnhancedSignalAgent',
    'PredictiveAnalysisAgent',
    'DeepLearningAgent',
    'StrategyTuner',
    'ModelRegistry',
    
    # Experimental Components
    'GrokBacktestCritic',
    'GrokSentimentAgent',
    'GrokStrategyAgent'
]

"""
GoldenSignalsAI Agent Framework - Directory Structure

agents/
├── core/                   # Core trading domains
│   ├── technical/         # Technical analysis agents
│   ├── fundamental/       # Fundamental analysis agents
│   ├── sentiment/         # Market sentiment & psychology
│   ├── risk/             # Risk management
│   ├── timing/           # Market timing & execution
│   └── portfolio/        # Portfolio management
├── infrastructure/         # System infrastructure
│   ├── data_sources/     # Data providers & aggregation
│   ├── integration/      # External system integration
│   └── monitoring/       # System & performance monitoring
├── research/              # Research & development
│   ├── backtesting/      # Backtesting framework
│   ├── ml/              # Machine learning models
│   └── optimization/     # Strategy optimization
├── common/                # Shared components
│   ├── base/            # Base classes & interfaces
│   ├── utils/           # Utility functions
│   └── registry/        # Agent registry & factory
└── experimental/          # Experimental features
    ├── grok/            # Grok-powered agents
    ├── vision/          # Computer vision agents
    └── adaptive/        # Adaptive learning agents
"""
