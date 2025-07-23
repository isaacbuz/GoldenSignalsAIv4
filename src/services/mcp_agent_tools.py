"""
MCP Agent Tools - Transform all agents into standardized MCP tools
This enables LLMs to discover and execute agent functions through a unified interface
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from services.mcp_tools import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolType,
    execute_mcp_tool,
    mcp_registry,
    mcp_tool,
)
from src.services.agent_registry import agent_registry
from src.services.redis_cache_service import redis_cache

logger = logging.getLogger(__name__)


# Extend ToolType for agent-specific categories
class AgentToolType(Enum):
    TECHNICAL_AGENT = "technical_agent"
    FORECAST_AGENT = "forecast_agent"
    SENTIMENT_AGENT = "sentiment_agent"
    CONSENSUS_AGENT = "consensus_agent"
    MARKET_AGENT = "market_agent"
    OPTIONS_AGENT = "options_agent"
    WORKFLOW_AGENT = "workflow_agent"


# Map agents to their tool types
AGENT_TYPE_MAPPING = {
    "RSIAgent": AgentToolType.TECHNICAL_AGENT,
    "MACDAgent": AgentToolType.TECHNICAL_AGENT,
    "VolumeAgent": AgentToolType.TECHNICAL_AGENT,
    "MomentumAgent": AgentToolType.TECHNICAL_AGENT,
    "PatternAgent": AgentToolType.TECHNICAL_AGENT,
    "SentimentAgent": AgentToolType.SENTIMENT_AGENT,
    "LSTMForecastAgent": AgentToolType.FORECAST_AGENT,
    "OptionsChainAgent": AgentToolType.OPTIONS_AGENT,
    "MarketRegimeAgent": AgentToolType.MARKET_AGENT,
}


@mcp_tool(AgentToolType.TECHNICAL_AGENT, "Analyze using any registered trading agent")
async def analyze_with_agent(
    agent_name: str, symbol: str, timeframe: str = "1h", parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute any registered agent's analysis through MCP

    Args:
        agent_name: Name of the agent (e.g., 'RSIAgent', 'MACDAgent')
        symbol: Trading symbol to analyze
        timeframe: Analysis timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        parameters: Agent-specific parameters

    Returns:
        Agent analysis results with confidence scores
    """
    try:
        # Check cache first
        cache_key = f"{agent_name}:{symbol}:{timeframe}"
        cached_result = await redis_cache.get_agent_analysis(agent_name, symbol, parameters)
        if cached_result:
            logger.info(f"Cache hit for {cache_key}")
            return cached_result

        # Get agent from registry
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found in registry")

        # Execute agent analysis
        logger.info(f"Executing {agent_name} analysis for {symbol}")
        result = await agent.analyze(symbol, timeframe=timeframe, **(parameters or {}))

        # Convert result to dict format
        analysis_result = {
            "agent": agent_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "signal": result.signal,
            "confidence": result.confidence,
            "analysis": result.analysis,
            "metadata": getattr(result, "metadata", {}),
            "success": True,
        }

        # Cache the result
        await redis_cache.set_agent_analysis(
            agent_name, symbol, analysis_result, ttl=300, params=parameters
        )

        return analysis_result

    except Exception as e:
        logger.error(f"Error executing {agent_name}: {str(e)}")
        return {"agent": agent_name, "symbol": symbol, "error": str(e), "success": False}


@mcp_tool(AgentToolType.CONSENSUS_AGENT, "Get weighted consensus from multiple agents")
async def get_agent_consensus(
    symbol: str, agents: List[str], timeframe: str = "1h", voting_method: str = "weighted"
) -> Dict[str, Any]:
    """
    Get consensus analysis from multiple agents with different voting methods

    Args:
        symbol: Trading symbol
        agents: List of agent names to consult
        timeframe: Analysis timeframe
        voting_method: 'weighted', 'majority', or 'unanimous'

    Returns:
        Consensus analysis with individual agent signals
    """
    try:
        # Execute all agents in parallel
        import asyncio

        agent_tasks = []

        for agent_name in agents:
            task = analyze_with_agent(agent_name, symbol, timeframe)
            agent_tasks.append(task)

        # Gather all results
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [
            r for r in agent_results if isinstance(r, dict) and r.get("success", False)
        ]

        if not successful_results:
            return {"success": False, "error": "No agents returned successful analysis"}

        # Calculate consensus based on voting method
        if voting_method == "weighted":
            consensus = _calculate_weighted_consensus(successful_results)
        elif voting_method == "majority":
            consensus = _calculate_majority_consensus(successful_results)
        else:  # unanimous
            consensus = _calculate_unanimous_consensus(successful_results)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "consensus": consensus,
            "voting_method": voting_method,
            "individual_signals": successful_results,
            "participating_agents": len(successful_results),
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error getting consensus: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp_tool(AgentToolType.FORECAST_AGENT, "Generate AI-enhanced price prediction")
async def predict_with_ai(
    symbol: str, horizon: int = 24, use_ensemble: bool = True, include_confidence_bands: bool = True
) -> Dict[str, Any]:
    """
    Generate AI-enhanced prediction using multiple models and agents

    Args:
        symbol: Trading symbol
        horizon: Prediction horizon in hours
        use_ensemble: Use ensemble of models
        include_confidence_bands: Include upper/lower confidence bands

    Returns:
        Prediction with trajectory and confidence bands
    """
    try:
        from src.services.advanced_ai_predictor import AdvancedAIPredictor

        # Get supporting agent analyses
        technical_agents = ["RSIAgent", "MACDAgent", "PatternAgent"]
        agent_insights = await get_agent_consensus(symbol, technical_agents)

        # Initialize predictor
        predictor = AdvancedAIPredictor()

        # Generate prediction with agent context
        prediction = await predictor.predict(
            symbol=symbol, horizon=horizon, agent_insights=agent_insights
        )

        result = {
            "symbol": symbol,
            "horizon_hours": horizon,
            "prediction": prediction["prediction"],
            "current_price": prediction["current_price"],
            "predicted_change_percent": prediction["predicted_change_percent"],
            "confidence_score": prediction["confidence_score"],
            "supporting_factors": prediction["factors"],
            "risk_score": prediction["risk_score"],
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }

        if include_confidence_bands:
            result["confidence_bands"] = {
                "upper": prediction["confidence_bounds"]["upper"],
                "lower": prediction["confidence_bounds"]["lower"],
            }

        return result

    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp_tool(AgentToolType.WORKFLOW_AGENT, "Execute complex multi-agent workflows")
async def execute_agent_workflow(
    workflow_name: str, symbol: str, parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute predefined multi-agent workflows for complex analysis

    Available workflows:
    - 'full_analysis': Complete technical + sentiment + forecast
    - 'risk_assessment': Options + market regime + volatility
    - 'entry_signal': Consensus for entry timing
    - 'exit_signal': Consensus for exit timing

    Args:
        workflow_name: Name of the workflow to execute
        symbol: Trading symbol
        parameters: Workflow-specific parameters

    Returns:
        Workflow execution results
    """
    workflows = {
        "full_analysis": {
            "agents": [
                "RSIAgent",
                "MACDAgent",
                "PatternAgent",
                "SentimentAgent",
                "LSTMForecastAgent",
            ],
            "steps": ["technical", "sentiment", "forecast", "consensus"],
        },
        "risk_assessment": {
            "agents": ["OptionsChainAgent", "MarketRegimeAgent", "VolumeAgent"],
            "steps": ["market_regime", "options_flow", "risk_calculation"],
        },
        "entry_signal": {
            "agents": ["RSIAgent", "MACDAgent", "MomentumAgent", "PatternAgent"],
            "steps": ["technical_consensus", "entry_validation"],
        },
        "exit_signal": {
            "agents": ["RSIAgent", "VolumeAgent", "MarketRegimeAgent"],
            "steps": ["exit_conditions", "risk_check"],
        },
    }

    if workflow_name not in workflows:
        return {
            "success": False,
            "error": f"Unknown workflow: {workflow_name}. Available: {list(workflows.keys())}",
        }

    try:
        workflow = workflows[workflow_name]
        results = {
            "workflow": workflow_name,
            "symbol": symbol,
            "steps": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Execute workflow steps
        if "technical" in workflow["steps"]:
            technical_agents = [
                a
                for a in workflow["agents"]
                if a in ["RSIAgent", "MACDAgent", "PatternAgent", "MomentumAgent"]
            ]
            results["steps"]["technical"] = await get_agent_consensus(symbol, technical_agents)

        if "sentiment" in workflow["steps"]:
            sentiment_result = await analyze_with_agent("SentimentAgent", symbol)
            results["steps"]["sentiment"] = sentiment_result

        if "forecast" in workflow["steps"]:
            forecast_result = await predict_with_ai(symbol, horizon=24)
            results["steps"]["forecast"] = forecast_result

        if "consensus" in workflow["steps"]:
            all_results = await get_agent_consensus(symbol, workflow["agents"])
            results["steps"]["consensus"] = all_results

        # Determine final recommendation
        results["recommendation"] = _determine_workflow_recommendation(results["steps"])
        results["success"] = True

        return results

    except Exception as e:
        logger.error(f"Error executing workflow {workflow_name}: {str(e)}")
        return {"success": False, "error": str(e), "workflow": workflow_name}


# Helper functions for consensus calculation
def _calculate_weighted_consensus(results: List[Dict]) -> Dict[str, Any]:
    """Calculate weighted consensus based on agent confidence scores"""
    total_weight = sum(r.get("confidence", 0.5) for r in results)

    buy_score = sum(r.get("confidence", 0.5) for r in results if r.get("signal") == "BUY")
    sell_score = sum(r.get("confidence", 0.5) for r in results if r.get("signal") == "SELL")

    if buy_score > sell_score and buy_score > total_weight * 0.6:
        action = "BUY"
    elif sell_score > buy_score and sell_score > total_weight * 0.6:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = max(buy_score, sell_score) / total_weight if total_weight > 0 else 0

    return {
        "action": action,
        "confidence": confidence,
        "buy_score": buy_score / total_weight if total_weight > 0 else 0,
        "sell_score": sell_score / total_weight if total_weight > 0 else 0,
        "method": "weighted",
    }


def _calculate_majority_consensus(results: List[Dict]) -> Dict[str, Any]:
    """Calculate simple majority consensus"""
    signals = [r.get("signal", "HOLD") for r in results]
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    total = len(signals)

    if buy_count > total / 2:
        action = "BUY"
        confidence = buy_count / total
    elif sell_count > total / 2:
        action = "SELL"
        confidence = sell_count / total
    else:
        action = "HOLD"
        confidence = 1 - (buy_count + sell_count) / total

    return {
        "action": action,
        "confidence": confidence,
        "buy_votes": buy_count,
        "sell_votes": sell_count,
        "hold_votes": total - buy_count - sell_count,
        "method": "majority",
    }


def _calculate_unanimous_consensus(results: List[Dict]) -> Dict[str, Any]:
    """Calculate unanimous consensus (all agents must agree)"""
    signals = [r.get("signal", "HOLD") for r in results]

    if all(s == "BUY" for s in signals):
        return {"action": "BUY", "confidence": 1.0, "method": "unanimous"}
    elif all(s == "SELL" for s in signals):
        return {"action": "SELL", "confidence": 1.0, "method": "unanimous"}
    else:
        # No unanimous decision
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "method": "unanimous",
            "reason": "No unanimous agreement",
        }


def _determine_workflow_recommendation(steps: Dict[str, Any]) -> Dict[str, Any]:
    """Determine final recommendation from workflow steps"""
    # Aggregate all signals
    all_signals = []

    for step_name, step_result in steps.items():
        if isinstance(step_result, dict):
            if "consensus" in step_result:
                all_signals.append(step_result["consensus"])
            elif "signal" in step_result:
                all_signals.append(
                    {
                        "action": step_result["signal"],
                        "confidence": step_result.get("confidence", 0.5),
                    }
                )

    if not all_signals:
        return {"action": "HOLD", "confidence": 0.0, "reason": "No clear signals"}

    # Weight by confidence
    buy_weight = sum(s.get("confidence", 0) for s in all_signals if s.get("action") == "BUY")
    sell_weight = sum(s.get("confidence", 0) for s in all_signals if s.get("action") == "SELL")

    if buy_weight > sell_weight * 1.2:  # Require 20% more confidence for action
        return {"action": "BUY", "confidence": buy_weight / len(all_signals)}
    elif sell_weight > buy_weight * 1.2:
        return {"action": "SELL", "confidence": sell_weight / len(all_signals)}
    else:
        return {"action": "HOLD", "confidence": 0.5}


# Auto-register all agents as individual MCP tools
def register_all_agents_as_mcp_tools():
    """Register each agent as an individual MCP tool for fine-grained access"""
    for agent_name, agent_class in agent_registry.get_all_agents().items():
        tool_type = AGENT_TYPE_MAPPING.get(agent_name, AgentToolType.TECHNICAL_AGENT)

        # Create a specific tool for this agent
        async def create_agent_specific_tool(symbol: str, timeframe: str = "1h", **kwargs):
            return await analyze_with_agent(agent_name, symbol, timeframe, kwargs)

        # Register with specific name
        tool_name = f"analyze_with_{agent_name.lower()}"

        # Create tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=f"Analyze market using {agent_name} - {agent_class.__doc__ or 'Specialized trading agent'}",
            type=tool_type,
            parameters=[
                ToolParameter("symbol", "string", "Trading symbol to analyze", required=True),
                ToolParameter(
                    "timeframe",
                    "string",
                    "Analysis timeframe (1m, 5m, 15m, 1h, 4h, 1d)",
                    required=False,
                    default="1h",
                ),
            ],
            returns=f"Analysis results from {agent_name}",
        )

        # Register the tool
        mcp_registry.register(tool_def, create_agent_specific_tool)
        logger.info(f"Registered MCP tool: {tool_name}")


# Register all tools on module import
register_all_agents_as_mcp_tools()

# Export key functions
__all__ = [
    "analyze_with_agent",
    "get_agent_consensus",
    "predict_with_ai",
    "execute_agent_workflow",
    "AgentToolType",
]
