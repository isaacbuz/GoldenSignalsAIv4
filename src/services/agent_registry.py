"""
Agent Registry Service
Centralized management for all trading agents
"""

import asyncio
import json
import logging
import os

# Import path setup
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.core.sentiment.sentiment_agent_unified import SentimentAgent
from agents.core.technical.macd_agent import MACDAgent

# Import all agents
from agents.core.technical.momentum.rsi_agent import RSIAgent
from agents.core.technical.pattern_agent_unified import PatternAgent
from agents.core.technical.volume.volume_agent import VolumeAgent
from agents.lstm_forecast_agent import LSTMForecastAgent
from agents.market_regime_classification_agent import MarketRegimeClassificationAgent
from agents.momentum import MomentumAgent
from agents.research.ml.options.options_chain_agent import OptionsChainAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """Metadata for registered agents"""

    name: str
    agent_class: Type
    category: str
    description: str
    weight: float
    enabled: bool = True
    performance_score: float = 0.5
    total_calls: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0


class AgentRegistry:
    """
    Central registry for all trading agents
    Provides discovery, instantiation, and performance tracking
    """

    def __init__(self):
        self._agents: Dict[str, AgentMetadata] = {}
        self._instances: Dict[str, Any] = {}
        self._performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Register all default agents"""
        # Technical Analysis Agents
        self.register(
            "rsi",
            RSIAgent,
            "technical",
            "Analyzes Relative Strength Index for overbought/oversold conditions",
            weight=1.2,
        )

        self.register(
            "macd",
            MACDAgent,
            "technical",
            "Tracks Moving Average Convergence Divergence for trend changes",
            weight=1.1,
        )

        self.register(
            "volume", VolumeAgent, "technical", "Monitors volume patterns and anomalies", weight=1.0
        )

        self.register(
            "momentum",
            MomentumAgent,
            "technical",
            "Measures price momentum and velocity",
            weight=1.15,
        )

        self.register(
            "pattern",
            PatternAgent,
            "technical",
            "Detects chart patterns (head & shoulders, triangles, etc.)",
            weight=1.3,
        )

        # Sentiment Analysis
        self.register(
            "sentiment",
            SentimentAgent,
            "sentiment",
            "Analyzes market sentiment from news and social data",
            weight=0.9,
        )

        # Machine Learning Agents
        self.register(
            "lstm_forecast",
            LSTMForecastAgent,
            "ml",
            "Uses LSTM neural networks for price predictions",
            weight=1.4,
        )

        self.register(
            "options_chain",
            OptionsChainAgent,
            "options",
            "Analyzes options flow and implied volatility",
            weight=1.25,
        )

        # Market Analysis
        self.register(
            "market_regime",
            MarketRegimeClassificationAgent,
            "market",
            "Detects market states (trending/ranging/volatile)",
            weight=1.0,
        )

        logger.info(f"Initialized {len(self._agents)} default agents")

    def register(
        self,
        name: str,
        agent_class: Type,
        category: str,
        description: str,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Register a new agent"""
        if name in self._agents:
            logger.warning(f"Agent {name} already registered, updating...")

        self._agents[name] = AgentMetadata(
            name=name,
            agent_class=agent_class,
            category=category,
            description=description,
            weight=weight,
            enabled=enabled,
        )

        logger.info(f"Registered agent: {name} ({category})")

    def get_agent(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get or create an agent instance"""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        metadata = self._agents[name]

        if not metadata.enabled:
            raise ValueError(f"Agent '{name}' is disabled")

        # Create instance if not exists
        if name not in self._instances:
            try:
                # Check if agent expects config parameter
                import inspect

                sig = inspect.signature(metadata.agent_class.__init__)
                if "config" in sig.parameters:
                    self._instances[name] = metadata.agent_class(config=config)
                else:
                    self._instances[name] = metadata.agent_class()

                logger.info(f"Created instance of agent: {name}")
            except Exception as e:
                logger.error(f"Failed to create agent {name}: {e}")
                raise

        return self._instances[name]

    def get_agents_by_category(self, category: str) -> List[str]:
        """Get all agents in a category"""
        return [
            name
            for name, metadata in self._agents.items()
            if metadata.category == category and metadata.enabled
        ]

    def get_all_agents(self, enabled_only: bool = True) -> List[str]:
        """Get all registered agents"""
        if enabled_only:
            return [name for name, metadata in self._agents.items() if metadata.enabled]
        return list(self._agents.keys())

    def get_agent_metadata(self, name: str) -> AgentMetadata:
        """Get metadata for an agent"""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")
        return self._agents[name]

    def update_agent_weight(self, name: str, weight: float) -> None:
        """Update agent weight for consensus"""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        self._agents[name].weight = weight
        logger.info(f"Updated weight for {name}: {weight}")

    def enable_agent(self, name: str) -> None:
        """Enable an agent"""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        self._agents[name].enabled = True
        logger.info(f"Enabled agent: {name}")

    def disable_agent(self, name: str) -> None:
        """Disable an agent"""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        self._agents[name].enabled = False
        # Remove instance to free memory
        if name in self._instances:
            del self._instances[name]
        logger.info(f"Disabled agent: {name}")

    async def analyze_with_agent(
        self, agent_name: str, symbol: str, market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze using a specific agent with performance tracking"""
        start_time = datetime.now()

        try:
            agent = self.get_agent(agent_name)

            # Call analyze method
            if hasattr(agent, "analyze"):
                result = await agent.analyze(symbol, market_data)
            else:
                # Fallback for agents without unified interface
                logger.warning(f"Agent {agent_name} doesn't have analyze method")
                result = {
                    "signal": "NEUTRAL",
                    "confidence": 0.5,
                    "reasoning": "Agent doesn't support unified interface",
                }

            # Track performance
            execution_time = (datetime.now() - start_time).total_seconds()
            self._track_performance(agent_name, True, execution_time)

            return result

        except Exception as e:
            logger.error(f"Agent {agent_name} analysis failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            self._track_performance(agent_name, False, execution_time)
            raise

    async def analyze_with_all_agents(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run analysis with all enabled agents"""
        agents_to_run = []

        if categories:
            for category in categories:
                agents_to_run.extend(self.get_agents_by_category(category))
        else:
            agents_to_run = self.get_all_agents(enabled_only=True)

        # Remove duplicates
        agents_to_run = list(set(agents_to_run))

        # Run all agents in parallel
        tasks = []
        for agent_name in agents_to_run:
            task = self.analyze_with_agent(agent_name, symbol, market_data)
            tasks.append((agent_name, task))

        results = {}
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                results[agent_name] = {"error": str(e), "signal": "NEUTRAL", "confidence": 0.0}

        return results

    def _track_performance(self, agent_name: str, success: bool, execution_time: float):
        """Track agent performance metrics"""
        if agent_name not in self._agents:
            return

        metadata = self._agents[agent_name]

        # Update call count
        metadata.total_calls += 1

        # Update success rate
        if success:
            metadata.success_rate = (
                metadata.success_rate * (metadata.total_calls - 1) + 1
            ) / metadata.total_calls
        else:
            metadata.success_rate = (
                metadata.success_rate * (metadata.total_calls - 1)
            ) / metadata.total_calls

        # Update average execution time
        metadata.avg_execution_time = (
            metadata.avg_execution_time * (metadata.total_calls - 1) + execution_time
        ) / metadata.total_calls

        # Add to history
        self._performance_history[agent_name].append(
            {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "execution_time": execution_time,
            }
        )

        # Keep only last 100 entries
        if len(self._performance_history[agent_name]) > 100:
            self._performance_history[agent_name] = self._performance_history[agent_name][-100:]

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all agents"""
        report = {}

        for name, metadata in self._agents.items():
            report[name] = {
                "category": metadata.category,
                "weight": metadata.weight,
                "enabled": metadata.enabled,
                "total_calls": metadata.total_calls,
                "success_rate": round(metadata.success_rate, 3),
                "avg_execution_time": round(metadata.avg_execution_time, 3),
                "performance_score": round(metadata.performance_score, 3),
            }

        return report

    def export_config(self) -> Dict[str, Any]:
        """Export registry configuration"""
        config = {"agents": {}, "version": "1.0"}

        for name, metadata in self._agents.items():
            config["agents"][name] = {
                "category": metadata.category,
                "description": metadata.description,
                "weight": metadata.weight,
                "enabled": metadata.enabled,
            }

        return config

    def import_config(self, config: Dict[str, Any]) -> None:
        """Import registry configuration"""
        if "agents" not in config:
            raise ValueError("Invalid config format")

        for name, settings in config["agents"].items():
            if name in self._agents:
                self._agents[name].weight = settings.get("weight", 1.0)
                self._agents[name].enabled = settings.get("enabled", True)
                logger.info(f"Updated config for agent: {name}")


# Global registry instance
agent_registry = AgentRegistry()


# Convenience functions
def get_agent(name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Get an agent instance"""
    return agent_registry.get_agent(name, config)


def get_all_agents(enabled_only: bool = True) -> List[str]:
    """Get list of all agents"""
    return agent_registry.get_all_agents(enabled_only)


def get_agents_by_category(category: str) -> List[str]:
    """Get agents in a specific category"""
    return agent_registry.get_agents_by_category(category)


# Export for use
__all__ = [
    "AgentRegistry",
    "agent_registry",
    "get_agent",
    "get_all_agents",
    "get_agents_by_category",
]
