"""
Trading Workflow using LangGraph
Implements a sophisticated state machine for multi-agent trading decisions
with proper observability and debugging capabilities
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from src.services.langsmith_observability import log_trading_decision, trace_agent

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market regime states"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class TradingState(TypedDict):
    """State for the trading workflow"""

    symbol: str
    current_price: float
    market_state: Optional[MarketState]
    agent_signals: Dict[str, Any]
    historical_insights: Optional[Dict[str, Any]]
    consensus: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    final_decision: Optional[Dict[str, Any]]
    timestamp: datetime
    messages: List[str]
    errors: List[str]


def add_message(state: TradingState, message: str) -> TradingState:
    """Helper to add messages to state"""
    state["messages"].append(f"[{datetime.now().isoformat()}] {message}")
    return state


async def detect_market_regime(state: TradingState) -> TradingState:
    """First node: Detect current market regime"""
    try:
        state = add_message(state, f"Detecting market regime for {state['symbol']}")

        # Import the actual market regime detection agent
        import os
        import sys

        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from agents.market_regime_classification_agent import (
            MarketRegimeClassificationAgent as MarketRegimeAgent,
        )

        agent = MarketRegimeAgent()

        # Trace the agent call
        @trace_agent("market_regime")
        async def analyze_regime(data):
            return await agent.analyze(data)

        result = await analyze_regime({"symbol": state["symbol"], "price": state["current_price"]})

        # Map agent result to MarketState enum
        regime_map = {
            "bullish": MarketState.TRENDING_UP,
            "bearish": MarketState.TRENDING_DOWN,
            "neutral": MarketState.RANGING,
            "volatile": MarketState.VOLATILE,
        }

        state["market_state"] = regime_map.get(
            result.get("regime", "").lower(), MarketState.UNKNOWN
        )

        state = add_message(state, f"Market regime detected: {state['market_state'].value}")

    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        state["errors"].append(str(e))
        state["market_state"] = MarketState.UNKNOWN

    return state


async def collect_agent_signals(state: TradingState) -> TradingState:
    """Second node: Collect signals from all agents"""
    try:
        state = add_message(state, f"Collecting agent signals for {state['symbol']}")

        # Use agent registry for centralized agent management
        from src.services.agent_registry import agent_registry

        # Get all enabled agents from registry
        agent_names = [
            "rsi",
            "macd",
            "sentiment",
            "volume",
            "momentum",
            "pattern",
            "lstm_forecast",
            "options_chain",
        ]

        # Prepare market data for agents
        market_data = {
            "symbol": state["symbol"],
            "current_price": state["current_price"],
            "market_state": state.get("market_state", {}),
        }

        # Collect signals in parallel using registry
        import asyncio

        tasks = []
        for agent_name in agent_names:
            try:
                # Skip disabled agents
                if not agent_registry.get_agent_metadata(agent_name).enabled:
                    continue

                task = agent_registry.analyze_with_agent(agent_name, state["symbol"], market_data)
                tasks.append((agent_name, task))
            except Exception as e:
                logger.warning(f"Skipping agent {agent_name}: {e}")

        # Execute all agents concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results
        state["agent_signals"] = {}
        for (agent_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                state["errors"].append(f"{agent_name}: {str(result)}")
                state["agent_signals"][agent_name] = {
                    "signal": "error",
                    "confidence": 0.0,
                    "error": str(result),
                }
            else:
                state["agent_signals"][agent_name] = result

        state = add_message(state, f"Collected signals from {len(state['agent_signals'])} agents")

    except Exception as e:
        logger.error(f"Error collecting agent signals: {e}")
        state["errors"].append(str(e))

    return state


async def search_similar_patterns(state: TradingState) -> TradingState:
    """Search vector memory for similar historical patterns"""
    try:
        state = add_message(state, "Searching for similar historical patterns")

        from src.services.vector_memory import find_similar_market_conditions, vector_memory

        # Get current market conditions
        current_conditions = {
            "trend": state["market_state"].value if state.get("market_state") else "unknown",
            "symbol": state["symbol"],
            "price": state["current_price"],
            "indicators": {},
        }

        # Add indicator values from agent signals if available
        for agent_name, signal in state.get("agent_signals", {}).items():
            if "indicator_value" in signal.get("metadata", {}):
                current_conditions["indicators"][agent_name] = signal["metadata"]["indicator_value"]

        # Search for similar conditions
        similar_conditions = await find_similar_market_conditions(
            current_conditions, symbol=state["symbol"], limit=5
        )

        # Analyze historical outcomes
        historical_insights = {
            "similar_setups_found": len(similar_conditions),
            "historical_success_rate": 0.0,
            "average_return": 0.0,
            "key_patterns": [],
        }

        if similar_conditions:
            successful = sum(
                1 for _, outcome in similar_conditions if outcome.get("profitable", False)
            )
            historical_insights["historical_success_rate"] = successful / len(similar_conditions)

            returns = [outcome.get("return_pct", 0) for _, outcome in similar_conditions]
            historical_insights["average_return"] = sum(returns) / len(returns) if returns else 0

            # Extract key patterns
            for memory, outcome in similar_conditions[:3]:
                historical_insights["key_patterns"].append(
                    {
                        "date": memory.timestamp.isoformat(),
                        "outcome": outcome.get("status", "unknown"),
                        "return": outcome.get("return_pct", 0),
                        "relevance": memory.relevance_score,
                    }
                )

        state["historical_insights"] = historical_insights

        state = add_message(
            state,
            f"Found {len(similar_conditions)} similar historical patterns with "
            f"{historical_insights['historical_success_rate']:.1%} success rate",
        )

    except Exception as e:
        logger.error(f"Error searching historical patterns: {e}")
        state["errors"].append(str(e))
        state["historical_insights"] = {"error": str(e)}

    return state


async def build_consensus(state: TradingState) -> TradingState:
    """Fourth node: Build consensus from agent signals (now includes historical context)"""
    try:
        state = add_message(state, "Building consensus from agent signals")

        signals = state["agent_signals"]

        if not signals:
            state["consensus"] = {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "No agent signals available",
            }
            return state

        # Filter out error signals
        valid_signals = {k: v for k, v in signals.items() if v.get("signal") != "error"}

        # Count votes with weights based on agent performance
        agent_weights = {
            "rsi_agent": 1.2,
            "macd_agent": 1.1,
            "sentiment_agent": 0.9,
            "volume_agent": 1.0,
            "momentum_agent": 1.15,
            "pattern_agent": 1.3,
            "lstm_agent": 1.4,
            "options_agent": 1.25,
        }

        weighted_votes = {"buy": 0, "sell": 0, "hold": 0}
        total_weight = 0

        for agent_name, signal in valid_signals.items():
            action = signal.get("signal", "hold").lower()
            confidence = signal.get("confidence", 0.5)
            weight = agent_weights.get(agent_name, 1.0)

            if action in weighted_votes:
                weighted_votes[action] += confidence * weight
                total_weight += weight

        # Determine consensus action
        if total_weight > 0:
            # Normalize votes
            for action in weighted_votes:
                weighted_votes[action] /= total_weight

            # Find dominant action
            dominant_action = max(weighted_votes, key=weighted_votes.get)
            consensus_confidence = weighted_votes[dominant_action]

            # Map to trading action
            action_map = {"buy": "BUY", "sell": "SELL", "hold": "HOLD"}

            state["consensus"] = {
                "action": action_map.get(dominant_action, "HOLD"),
                "confidence": consensus_confidence,
                "weighted_votes": weighted_votes,
                "agents_total": len(valid_signals),
                "agents_error": len(signals) - len(valid_signals),
            }
        else:
            state["consensus"] = {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "No valid signals to build consensus",
            }

        state = add_message(
            state,
            f"Consensus: {state['consensus']['action']} "
            f"(confidence: {state['consensus']['confidence']:.2%})",
        )

    except Exception as e:
        logger.error(f"Error building consensus: {e}")
        state["errors"].append(str(e))
        state["consensus"] = {"action": "HOLD", "confidence": 0.0, "error": str(e)}

    return state


async def assess_risk(state: TradingState) -> TradingState:
    """Fourth node: Assess risk based on market state and consensus"""
    try:
        state = add_message(state, "Assessing risk")

        market_state = state.get("market_state", MarketState.UNKNOWN)
        consensus = state.get("consensus", {})

        # Base risk score
        risk_score = 0.5

        # Market state risk adjustments
        market_risk_factors = {
            MarketState.VOLATILE: 0.3,
            MarketState.TRENDING_UP: -0.1,
            MarketState.TRENDING_DOWN: -0.1,
            MarketState.RANGING: 0.1,
            MarketState.UNKNOWN: 0.2,
        }

        risk_score += market_risk_factors.get(market_state, 0)

        # Consensus confidence adjustments
        confidence = consensus.get("confidence", 0.5)
        if confidence > 0.8:
            risk_score -= 0.15
        elif confidence < 0.5:
            risk_score += 0.2

        # Action-specific risk adjustments
        action = consensus.get("action", "HOLD")
        if action == "BUY" and market_state == MarketState.TRENDING_DOWN:
            risk_score += 0.2
        elif action == "SELL" and market_state == MarketState.TRENDING_UP:
            risk_score += 0.2

        # Clamp risk score
        risk_score = max(0.1, min(0.9, risk_score))

        # Calculate position sizing and risk parameters
        position_size = (1.0 - risk_score) * confidence
        stop_loss_pct = 0.02 + (risk_score * 0.03)  # 2-5% stop loss
        take_profit_pct = 0.05 + ((1 - risk_score) * 0.05)  # 5-10% take profit

        state["risk_assessment"] = {
            "risk_score": risk_score,
            "position_size": position_size,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "max_exposure": 0.1 * (1 - risk_score),  # Max 10% portfolio exposure
            "risk_reward_ratio": take_profit_pct / stop_loss_pct,
        }

        state = add_message(
            state,
            f"Risk assessment complete: score={risk_score:.2f}, " f"position={position_size:.2%}",
        )

    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        state["errors"].append(str(e))
        state["risk_assessment"] = {"risk_score": 0.9, "position_size": 0.0, "error": str(e)}

    return state


async def make_final_decision(state: TradingState) -> TradingState:
    """Final node: Make trading decision"""
    try:
        state = add_message(state, "Making final trading decision")

        consensus = state.get("consensus", {})
        risk = state.get("risk_assessment", {})

        # Decision criteria
        min_confidence = 0.65
        max_risk_score = 0.7
        min_risk_reward = 1.5

        # Check if we should execute
        should_execute = (
            consensus.get("action", "HOLD") != "HOLD"
            and consensus.get("confidence", 0) >= min_confidence
            and risk.get("risk_score", 1.0) <= max_risk_score
            and risk.get("risk_reward_ratio", 0) >= min_risk_reward
            and len(state.get("errors", [])) == 0
        )

        if should_execute:
            action = consensus["action"]
            position_size = risk["position_size"]
            current_price = state["current_price"]

            state["final_decision"] = {
                "execute": True,
                "action": action,
                "confidence": consensus["confidence"] * (1 - risk["risk_score"]),
                "position_size": position_size,
                "entry_price": current_price,
                "stop_loss": current_price * (1 - risk["stop_loss_pct"])
                if action == "BUY"
                else current_price * (1 + risk["stop_loss_pct"]),
                "take_profit": current_price * (1 + risk["take_profit_pct"])
                if action == "BUY"
                else current_price * (1 - risk["take_profit_pct"]),
                "max_exposure": risk["max_exposure"],
                "reasoning": _generate_reasoning(state),
                "execution_time": datetime.now().isoformat(),
                "risk_level": "HIGH"
                if risk["risk_score"] > 0.7
                else "MEDIUM"
                if risk["risk_score"] > 0.4
                else "LOW",
            }

            # Validate decision with Guardrails
            try:
                from src.services.guardrails_validation import validate_trading_decision

                validation_context = {
                    "account_value": 100000,  # This would come from actual account data
                    "current_price": state["current_price"],
                    "stop_loss": state["final_decision"]["stop_loss"],
                }

                validation_result = await validate_trading_decision(
                    state["final_decision"], validation_context
                )

                if validation_result["valid"]:
                    if validation_result.get("modifications"):
                        state = add_message(state, "Decision modified by Guardrails validation")
                    state["final_decision"] = validation_result["decision"]
                else:
                    # Validation failed - default to HOLD
                    logger.error(f"Guardrails validation failed: {validation_result.get('errors')}")
                    state["final_decision"] = {
                        "execute": False,
                        "action": "HOLD",
                        "confidence": 0.0,
                        "reasoning": f"Validation failed: {validation_result.get('errors')}",
                        "risk_level": "HIGH",
                    }
                    state = add_message(state, "Decision failed validation - defaulting to HOLD")

            except Exception as e:
                logger.error(f"Guardrails validation error: {e}")
                state["errors"].append(f"Validation error: {str(e)}")
        else:
            # Generate reason for not executing
            reasons = []
            if consensus.get("action", "HOLD") == "HOLD":
                reasons.append("Consensus is HOLD")
            if consensus.get("confidence", 0) < min_confidence:
                reasons.append(f"Low confidence ({consensus.get('confidence', 0):.2%})")
            if risk.get("risk_score", 1.0) > max_risk_score:
                reasons.append(f"High risk ({risk.get('risk_score', 1.0):.2f})")
            if risk.get("risk_reward_ratio", 0) < min_risk_reward:
                reasons.append(f"Poor risk/reward ({risk.get('risk_reward_ratio', 0):.2f})")
            if state.get("errors"):
                reasons.append(f"{len(state['errors'])} errors occurred")

            state["final_decision"] = {
                "execute": False,
                "action": "HOLD",
                "confidence": 0.0,
                "position_size": 0.0,
                "reasoning": "; ".join(reasons),
                "execution_time": datetime.now().isoformat(),
            }

        state = add_message(
            state,
            f"Decision: {state['final_decision']['action']} "
            f"(execute: {state['final_decision']['execute']})",
        )

        # Store decision in vector memory for future reference
        try:
            from src.services.vector_memory import remember_trading_decision

            # Get agent signals for memory
            agents_involved = list(state.get("agent_signals", {}).keys())

            # Store the decision
            await remember_trading_decision(
                symbol=state["symbol"],
                decision=state["final_decision"],
                market_context={
                    "market_state": state["market_state"].value,
                    "current_price": state["current_price"],
                    "consensus": state["consensus"],
                    "risk_assessment": state["risk_assessment"],
                },
                agents_involved=agents_involved,
            )

            state = add_message(state, "Decision stored in vector memory")
        except Exception as e:
            logger.warning(f"Failed to store decision in memory: {e}")

        # Log to LangSmith for observability
        try:
            await log_trading_decision(
                symbol=state["symbol"],
                decision=state["final_decision"],
                context={
                    "market_state": state.get("market_state", MarketState.UNKNOWN).value,
                    "agent_signals": state.get("agent_signals", {}),
                    "historical_insights": state.get("historical_insights", {}),
                    "consensus": state.get("consensus", {}),
                    "risk_assessment": state.get("risk_assessment", {}),
                },
            )
            state = add_message(state, "Decision logged to LangSmith")
        except Exception as e:
            logger.warning(f"Failed to log to LangSmith: {e}")

    except Exception as e:
        logger.error(f"Error making final decision: {e}")
        state["errors"].append(str(e))
        state["final_decision"] = {"execute": False, "action": "HOLD", "error": str(e)}

    return state


def _generate_reasoning(state: TradingState) -> str:
    """Generate human-readable reasoning for the decision"""
    consensus = state.get("consensus", {})
    risk = state.get("risk_assessment", {})
    market = state.get("market_state", MarketState.UNKNOWN)

    # Build reasoning
    parts = []

    # Market context
    parts.append(f"Market is {market.value.replace('_', ' ')}")

    # Agent consensus
    if "weighted_votes" in consensus:
        votes = consensus["weighted_votes"]
        parts.append(
            f"Agents voted: BUY={votes.get('buy', 0):.1%}, "
            f"SELL={votes.get('sell', 0):.1%}, "
            f"HOLD={votes.get('hold', 0):.1%}"
        )

    # Confidence and risk
    parts.append(
        f"Consensus: {consensus.get('action', 'HOLD')} with "
        f"{consensus.get('confidence', 0):.1%} confidence"
    )
    parts.append(f"Risk score: {risk.get('risk_score', 0):.1%}")
    parts.append(f"Risk/Reward: {risk.get('risk_reward_ratio', 0):.2f}:1")

    # Agent performance
    if state.get("agent_signals"):
        performing_agents = [
            name
            for name, signal in state["agent_signals"].items()
            if signal.get("confidence", 0) > 0.8 and signal.get("signal") != "error"
        ]
        if performing_agents:
            parts.append(f"High confidence from: {', '.join(performing_agents)}")

    return ". ".join(parts)


def should_continue(state: TradingState) -> Literal["assess_risk", "end"]:
    """Conditional edge to determine if we should continue after consensus"""
    consensus = state.get("consensus", {})

    # Skip risk assessment if consensus is too weak or action is HOLD
    if consensus.get("action") == "HOLD" or consensus.get("confidence", 0) < 0.5:
        return "end"

    return "assess_risk"


class TradingWorkflowLangGraph:
    """
    LangGraph-based trading workflow with full observability
    """

    def __init__(self):
        # Initialize the graph
        self.workflow = StateGraph(TradingState)

        # Add nodes
        self.workflow.add_node("detect_market_regime", detect_market_regime)
        self.workflow.add_node("collect_agent_signals", collect_agent_signals)
        self.workflow.add_node("search_similar_patterns", search_similar_patterns)
        self.workflow.add_node("build_consensus", build_consensus)
        self.workflow.add_node("assess_risk", assess_risk)
        self.workflow.add_node("make_final_decision", make_final_decision)

        # Add edges
        self.workflow.add_edge("detect_market_regime", "collect_agent_signals")
        self.workflow.add_edge("collect_agent_signals", "search_similar_patterns")
        self.workflow.add_edge("search_similar_patterns", "build_consensus")

        # Conditional edge after consensus
        self.workflow.add_conditional_edges(
            "build_consensus", should_continue, {"assess_risk": "assess_risk", "end": END}
        )

        self.workflow.add_edge("assess_risk", "make_final_decision")
        self.workflow.add_edge("make_final_decision", END)

        # Set entry point
        self.workflow.set_entry_point("detect_market_regime")

        # Compile with checkpointer for observability
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def execute_workflow(self, symbol: str, price: float) -> Dict[str, Any]:
        """Execute the complete workflow"""
        # Initialize state
        initial_state = TradingState(
            symbol=symbol,
            current_price=price,
            market_state=None,
            agent_signals={},
            historical_insights=None,
            consensus=None,
            risk_assessment=None,
            final_decision=None,
            timestamp=datetime.now(),
            messages=[],
            errors=[],
        )

        # Run workflow with thread_id for tracking
        thread_id = f"{symbol}_{datetime.now().isoformat()}"

        try:
            # Execute the graph
            result = await self.app.ainvoke(
                initial_state, config={"configurable": {"thread_id": thread_id}}
            )

            # Log execution path
            logger.info(f"Workflow completed for {symbol}")
            logger.info(f"Messages: {json.dumps(result.get('messages', []), indent=2)}")

            if result.get("errors"):
                logger.warning(f"Errors occurred: {result['errors']}")

            return result.get(
                "final_decision",
                {
                    "execute": False,
                    "action": "HOLD",
                    "error": "Workflow did not produce a decision",
                },
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {"execute": False, "action": "HOLD", "error": str(e)}

    def get_workflow_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get execution history for debugging"""
        return list(self.checkpointer.get(thread_id))

    def visualize_workflow(self) -> str:
        """Generate workflow visualization"""
        try:
            # This returns a mermaid diagram
            return self.app.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            return "Unable to generate visualization"


# Singleton instance
trading_workflow = TradingWorkflowLangGraph()


async def run_trading_analysis(symbol: str, price: float) -> Dict[str, Any]:
    """Run the LangGraph trading workflow for a symbol"""
    return await trading_workflow.execute_workflow(symbol, price)


def get_workflow_visualization() -> str:
    """Get workflow visualization for debugging"""
    return trading_workflow.visualize_workflow()
