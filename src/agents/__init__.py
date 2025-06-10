"""
Advanced Multi-Agent Trading System for GoldenSignalsAI V3

This module implements a sophisticated agentic architecture combining:
- CrewAI for agent coordination and task management
- Specialized trading agents for different market aspects
- Real-time signal fusion and consensus mechanism
- Self-improving AI through reinforcement learning feedback
"""

from .orchestrator import AgentOrchestrator
from .technical_analysis import TechnicalAnalysisAgent
from .sentiment_analysis import SentimentAnalysisAgent
from .momentum import MomentumAgent
from .mean_reversion import MeanReversionAgent
from .volume_analysis import VolumeAnalysisAgent

__all__ = [
    "AgentOrchestrator",
    "TechnicalAnalysisAgent", 
    "SentimentAnalysisAgent",
    "MomentumAgent",
    "MeanReversionAgent",
    "VolumeAnalysisAgent"
] 