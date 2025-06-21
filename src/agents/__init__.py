"""
Advanced Multi-Agent Trading System for GoldenSignalsAI V3

This module implements a sophisticated agentic architecture combining:
- CrewAI for agent coordination and task management
- Specialized trading agents for different market aspects
- Real-time signal fusion and consensus mechanism
- Self-improving AI through reinforcement learning feedback
"""

from src.agents.orchestrator import AgentOrchestrator
from src.agents.base import BaseAgent
from src.agents.technical_analysis import TechnicalAnalysisAgent
from src.agents.sentiment_analysis import SentimentAnalysisAgent
from src.agents.volume_analysis import VolumeAnalysisAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.momentum import MomentumAgent
from src.agents.common.base import AgentRegistry, registry
from src.agents.common.models import MarketData, Signal, Prediction
from src.agents.transformer import (
    TransformerAgent,
    TransformerModel,
    TransformerConfig,
    get_transformer_device,
    prepare_transformer_input,
    calculate_attention_weights
)

__all__ = [
    'AgentOrchestrator',
    'BaseAgent',
    'TechnicalAnalysisAgent',
    'SentimentAnalysisAgent',
    'VolumeAnalysisAgent',
    'MeanReversionAgent',
    'MomentumAgent',
    'AgentRegistry',
    'registry',
    'MarketData',
    'Signal',
    'Prediction',
    'TransformerAgent',
    'TransformerModel',
    'TransformerConfig',
    'get_transformer_device',
    'prepare_transformer_input',
    'calculate_attention_weights'
] 