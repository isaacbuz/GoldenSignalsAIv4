"""
Unified Base Agent for GoldenSignalsAI
Provides standardized interface and MCP tool integration for all agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from enum import Enum

# Import MCP tools for standardized access
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.mcp_tools import execute_mcp_tool, get_available_tools

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Standardized signal strengths"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    @classmethod
    def from_score(cls, score: float) -> 'SignalStrength':
        """Convert numeric score (-1 to 1) to signal strength"""
        if score >= 0.5:
            return cls.STRONG_BUY
        elif score >= 0.2:
            return cls.BUY
        elif score >= -0.2:
            return cls.NEUTRAL
        elif score >= -0.5:
            return cls.SELL
        else:
            return cls.STRONG_SELL


@dataclass
class AgentSignal:
    """Standardized agent signal output"""
    agent_name: str
    signal: SignalStrength
    confidence: float  # 0.0 to 1.0
    reasoning: str
    data: Dict[str, Any]
    timestamp: datetime
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms
        }


class UnifiedBaseAgent(ABC):
    """
    Unified base class for all trading agents
    Provides MCP tool integration and standardized interfaces
    """

    def __init__(self, name: str, weight: float = 1.0, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent

        Args:
            name: Agent name for identification
            weight: Agent weight in consensus (default 1.0)
            config: Optional configuration dictionary
        """
        self.name = name
        self.weight = weight
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._performance_history: List[Tuple[float, float]] = []  # (prediction, actual)

    @abstractmethod
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core analysis method that each agent must implement

        Args:
            market_data: Dictionary containing at minimum:
                - symbol: str
                - current_price: float
                - historical_data: List[Dict] (OHLCV data)

        Returns:
            Dictionary with analysis results specific to the agent
        """
        pass

    @abstractmethod
    def get_required_data_fields(self) -> List[str]:
        """
        Return list of required fields in market_data
        This helps with data validation and preparation
        """
        pass

    async def analyze(self, symbol: str, market_data: Optional[Dict[str, Any]] = None) -> AgentSignal:
        """
        Standardized analyze method that wraps agent-specific logic

        Args:
            symbol: Trading symbol to analyze
            market_data: Optional pre-fetched market data

        Returns:
            AgentSignal with standardized format
        """
        start_time = datetime.now()

        try:
            # Fetch market data if not provided
            if market_data is None:
                market_data = await self._fetch_market_data(symbol)
            else:
                market_data["symbol"] = symbol

            # Validate required fields
            self._validate_market_data(market_data)

            # Perform agent-specific analysis
            analysis_result = await self.analyze_market(market_data)

            # Convert to standardized signal
            signal = self._process_analysis_result(analysis_result)

            # Calculate execution time
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return AgentSignal(
                agent_name=self.name,
                signal=signal["signal"],
                confidence=signal["confidence"],
                reasoning=signal["reasoning"],
                data=signal.get("data", {}),
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {e}")
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return AgentSignal(
                agent_name=self.name,
                signal=SignalStrength.NEUTRAL,
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                data={"error": str(e)},
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )

    def _validate_market_data(self, market_data: Dict[str, Any]) -> None:
        """Validate that market data contains required fields"""
        required_fields = self.get_required_data_fields()
        missing_fields = [field for field in required_fields if field not in market_data]

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    def _process_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent-specific result into standardized format"""
        # Handle different result formats from legacy agents

        # Try to extract signal
        if "signal" in result:
            if isinstance(result["signal"], str):
                signal = self._parse_signal_string(result["signal"])
            else:
                signal = SignalStrength.from_score(float(result["signal"]))
        elif "action" in result:
            signal = self._parse_signal_string(result["action"])
        elif "recommendation" in result:
            signal = self._parse_signal_string(result["recommendation"])
        else:
            # Try to infer from score/sentiment
            score = result.get("score", result.get("sentiment", 0))
            signal = SignalStrength.from_score(float(score))

        # Extract confidence
        confidence = float(result.get("confidence", result.get("probability", 0.5)))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Extract reasoning
        reasoning = result.get("reasoning", result.get("analysis", result.get("explanation", "")))
        if not reasoning:
            reasoning = f"{self.name} analysis completed"

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "data": result
        }

    def _parse_signal_string(self, signal_str: str) -> SignalStrength:
        """Parse string signal to enum"""
        signal_map = {
            "STRONG_BUY": SignalStrength.STRONG_BUY,
            "BUY": SignalStrength.BUY,
            "HOLD": SignalStrength.NEUTRAL,
            "NEUTRAL": SignalStrength.NEUTRAL,
            "SELL": SignalStrength.SELL,
            "STRONG_SELL": SignalStrength.STRONG_SELL,
        }
        return signal_map.get(signal_str.upper(), SignalStrength.NEUTRAL)

    # MCP Tool Integration Methods

    async def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data using MCP tools"""
        try:
            result = await execute_mcp_tool("get_market_data", symbol=symbol)
            if result.success:
                return result.data
            else:
                raise Exception(f"Failed to fetch market data: {result.error}")
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            # Return minimal data structure
            return {
                "symbol": symbol,
                "current_price": 0.0,
                "historical_data": []
            }

    async def get_technical_indicators(self, symbol: str, indicators: List[str]) -> Dict[str, Any]:
        """Get technical indicators using MCP tools"""
        try:
            result = await execute_mcp_tool(
                "calculate_technicals",
                symbol=symbol,
                indicators=indicators
            )
            if result.success:
                return result.data
            else:
                self.logger.warning(f"Failed to get indicators: {result.error}")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting technical indicators: {e}")
            return {}

    async def get_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis using MCP tools"""
        try:
            result = await execute_mcp_tool("analyze_sentiment", symbol=symbol)
            if result.success:
                return result.data
            else:
                self.logger.warning(f"Failed to get sentiment: {result.error}")
                return {"sentiment": 0, "confidence": 0}
        except Exception as e:
            self.logger.error(f"Error getting sentiment: {e}")
            return {"sentiment": 0, "confidence": 0}

    async def get_options_flow(self, symbol: str) -> Dict[str, Any]:
        """Get options flow analysis using MCP tools"""
        try:
            result = await execute_mcp_tool("analyze_options_flow", symbol=symbol)
            if result.success:
                return result.data
            else:
                self.logger.warning(f"Failed to get options flow: {result.error}")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting options flow: {e}")
            return {}

    async def calculate_risk_metrics(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        position_size: int,
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Calculate risk metrics using MCP tools"""
        try:
            result = await execute_mcp_tool(
                "calculate_risk",
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                position_size=position_size,
                portfolio_value=portfolio_value
            )
            if result.success:
                return result.data
            else:
                self.logger.warning(f"Failed to calculate risk: {result.error}")
                return {}
        except Exception as e:
            self.logger.error(f"Error calculating risk: {e}")
            return {}

    # Performance Tracking Methods

    def update_performance(self, prediction: float, actual: float) -> None:
        """Update agent performance history"""
        self._performance_history.append((prediction, actual))
        # Keep only last 100 predictions
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]

    def get_accuracy(self) -> float:
        """Calculate agent accuracy based on recent predictions"""
        if not self._performance_history:
            return 0.5  # Default to 50% if no history

        correct = 0
        for prediction, actual in self._performance_history:
            # Consider prediction correct if direction matches
            if (prediction > 0 and actual > 0) or (prediction < 0 and actual < 0):
                correct += 1

        return correct / len(self._performance_history)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        if not self._performance_history:
            return {
                "accuracy": 0.5,
                "avg_confidence": 0.5,
                "total_predictions": 0
            }

        return {
            "accuracy": self.get_accuracy(),
            "avg_confidence": sum(abs(p) for p, _ in self._performance_history) / len(self._performance_history),
            "total_predictions": len(self._performance_history)
        }


# Export classes
__all__ = ['UnifiedBaseAgent', 'AgentSignal', 'SignalStrength']
