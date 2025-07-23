"""
Golden Eye Orchestrator - The brain that connects LLMs, MCP tools, and the UI
Handles query routing, LLM specialization, and live execution
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from anthropic import Anthropic

from src.services.ai_orchestrator import AIOrchestrator, TaskType
from src.services.langsmith_observability import trace_llm_call, trace_workflow
from src.services.mcp_agent_tools import (
    analyze_with_agent,
    execute_agent_workflow,
    get_agent_consensus,
    predict_with_ai,
)
from src.services.mcp_tools import (
    execute_mcp_tool,
    get_all_mcp_tools,
    get_mcp_tools_for_anthropic,
    get_mcp_tools_for_openai,
)
from src.services.redis_cache_service import redis_cache

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of queries the Golden Eye can handle"""

    PREDICTION = "prediction"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_SENTIMENT = "market_sentiment"
    ENTRY_EXIT_SIGNAL = "entry_exit_signal"
    GENERAL_QUESTION = "general_question"
    WORKFLOW_EXECUTION = "workflow_execution"
    LIVE_CHART_ACTION = "live_chart_action"


@dataclass
class GoldenEyeContext:
    """Context for Golden Eye query processing"""

    symbol: str
    timeframe: str = "1h"
    user_id: Optional[str] = None
    chart_visible: bool = True
    available_agents: List[str] = None
    preferences: Dict[str, Any] = None


class GoldenEyeOrchestrator:
    """
    Master orchestrator that connects Golden Eye Chat to LLMs and MCP tools
    Provides intelligent routing, streaming responses, and live execution
    """

    def __init__(self):
        self.ai_orchestrator = AIOrchestrator()
        self._intent_classifier = IntentClassifier()
        self._llm_router = LLMRouter()
        self._execution_sandbox = ExecutionSandbox()
        self._tool_discovery = ToolDiscovery()

    @trace_workflow("golden_eye_query")
    async def process_query(
        self, query: str, context: GoldenEyeContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a Golden Eye chat query with streaming response

        Yields events:
        - thinking: Shows what the system is doing
        - tool_execution: When MCP tools are called
        - text: LLM response text
        - chart_action: Actions to perform on the chart
        - error: If something goes wrong
        """
        try:
            # 1. Classify intent
            intent = await self._intent_classifier.classify(query)
            logger.info(f"Classified intent: {intent}")

            yield {
                "type": "thinking",
                "message": f"Understanding your {intent.value.replace('_', ' ')} request...",
                "intent": intent.value,
            }

            # 2. Extract entities and parameters
            entities = await self._extract_entities(query, context)

            # 3. Route to appropriate LLM with specialized tools
            llm_config = self._llm_router.route(intent, entities)

            yield {
                "type": "thinking",
                "message": f"Consulting {llm_config['llm_name']} for {llm_config['specialty']}...",
                "llm": llm_config["llm"],
                "agents": llm_config["agents"],
            }

            # 4. Process with appropriate handler
            handler = self._get_intent_handler(intent)
            async for event in handler(query, entities, llm_config, context):
                yield event

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            yield {"type": "error", "message": f"I encountered an error: {str(e)}", "error": str(e)}

    async def _handle_prediction_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle prediction-related queries"""
        symbol = entities.get("symbol", context.symbol)
        horizon = entities.get("horizon", 24)

        # First, get AI prediction using MCP tool
        yield {"type": "thinking", "message": f"Generating AI prediction for {symbol}..."}

        prediction_result = await predict_with_ai(symbol=symbol, horizon=horizon, use_ensemble=True)

        yield {"type": "tool_execution", "tool": "predict_with_ai", "result": prediction_result}

        # Get LLM to explain the prediction
        messages = [
            {
                "role": "system",
                "content": f"You are {llm_config['llm_name']}, specialized in {llm_config['specialty']}. Explain this prediction clearly.",
            },
            {
                "role": "user",
                "content": f"""Explain this prediction for {symbol}:

Prediction: {json.dumps(prediction_result, indent=2)}

User Query: {query}

Provide a clear, actionable explanation focusing on:
1. What the prediction means
2. Key factors driving it
3. Risk considerations
4. Suggested actions""",
            },
        ]

        # Stream LLM response
        async for chunk in self._stream_llm_response(llm_config["llm"], messages):
            yield chunk

        # Generate chart action
        if prediction_result.get("success") and context.chart_visible:
            yield {
                "type": "chart_action",
                "action": {
                    "type": "draw_prediction",
                    "symbol": symbol,
                    "prediction": prediction_result["prediction"],
                    "confidence_bands": prediction_result.get("confidence_bands", {}),
                    "horizon": horizon,
                },
            }

    async def _handle_technical_analysis_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle technical analysis queries"""
        symbol = entities.get("symbol", context.symbol)
        agents = entities.get("agents", llm_config["agents"])

        # Get consensus from technical agents
        yield {"type": "thinking", "message": f"Consulting {len(agents)} technical agents..."}

        consensus_result = await get_agent_consensus(
            symbol=symbol, agents=agents, timeframe=context.timeframe, voting_method="weighted"
        )

        yield {
            "type": "tool_execution",
            "tool": "get_agent_consensus",
            "result": consensus_result,
            "agents_consulted": agents,
        }

        # Have LLM interpret the results
        messages = [
            {
                "role": "system",
                "content": f"You are {llm_config['llm_name']}. Interpret these technical analysis results.",
            },
            {
                "role": "user",
                "content": f"""Analyze these technical indicators for {symbol}:

{json.dumps(consensus_result, indent=2)}

Query: {query}

Provide:
1. Current technical setup
2. Key levels to watch
3. Trade recommendation
4. Risk/reward analysis""",
            },
        ]

        async for chunk in self._stream_llm_response(
            llm_config["llm"], messages, tools=llm_config["tools"]
        ):
            yield chunk

        # Add signals to chart
        if consensus_result.get("success") and context.chart_visible:
            yield {
                "type": "chart_action",
                "action": {
                    "type": "add_agent_signals",
                    "signals": consensus_result["individual_signals"],
                    "consensus": consensus_result["consensus"],
                },
            }

    async def _handle_workflow_execution(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex workflow execution"""
        workflow_name = entities.get("workflow", "full_analysis")
        symbol = entities.get("symbol", context.symbol)

        yield {"type": "thinking", "message": f"Executing {workflow_name} workflow..."}

        # Execute workflow
        workflow_result = await execute_agent_workflow(
            workflow_name=workflow_name, symbol=symbol, parameters=entities.get("parameters", {})
        )

        yield {
            "type": "tool_execution",
            "tool": "execute_agent_workflow",
            "workflow": workflow_name,
            "result": workflow_result,
        }

        # Have LLM synthesize the results
        messages = [
            {
                "role": "system",
                "content": "You are an expert trading analyst. Synthesize these workflow results into actionable insights.",
            },
            {
                "role": "user",
                "content": f"""Workflow '{workflow_name}' completed for {symbol}:

{json.dumps(workflow_result, indent=2)}

Provide a comprehensive summary with:
1. Key findings
2. Action items
3. Risk factors
4. Timeline for execution""",
            },
        ]

        async for chunk in self._stream_llm_response(llm_config["llm"], messages):
            yield chunk

    async def _stream_llm_response(
        self, llm: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from LLM with tool calling support"""
        try:
            if llm == "openai":
                async for chunk in self._stream_openai(messages, tools):
                    yield chunk
            elif llm == "anthropic":
                async for chunk in self._stream_claude(messages, tools):
                    yield chunk
            elif llm == "grok":
                async for chunk in self._stream_grok(messages, tools):
                    yield chunk
            else:
                yield {"type": "error", "message": f"Unknown LLM: {llm}"}
        except Exception as e:
            logger.error(f"LLM streaming error: {str(e)}")
            yield {"type": "error", "message": f"LLM error: {str(e)}"}

    @trace_llm_call("openai", "gpt-4o")
    async def _stream_openai(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from OpenAI with function calling"""
        client = self.ai_orchestrator.providers["openai"].client

        params = {"model": "gpt-4o", "messages": messages, "stream": True, "temperature": 0.7}

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        stream = await asyncio.to_thread(client.chat.completions.create, **params)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "text", "content": chunk.choices[0].delta.content}

            # Handle tool calls
            if chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.function:
                        # Execute the tool
                        tool_result = await execute_mcp_tool(
                            tool_call.function.name, **json.loads(tool_call.function.arguments)
                        )
                        yield {
                            "type": "tool_execution",
                            "tool": tool_call.function.name,
                            "result": tool_result.data,
                        }

    async def _stream_claude(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Claude with tool support"""
        client = self.ai_orchestrator.providers["anthropic"].client

        # Claude uses a different message format
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Claude puts system in the first user message
                claude_messages.append(
                    {"role": "user", "content": f"System: {msg['content']}\n\nUser: "}
                )
            else:
                claude_messages.append(msg)

        params = {
            "model": "claude-3-opus-20240229",
            "messages": claude_messages,
            "max_tokens": 4000,
            "temperature": 0.7,
            "stream": True,
        }

        if tools:
            params["tools"] = tools

        stream = await asyncio.to_thread(client.messages.create, **params)

        for chunk in stream:
            if chunk.type == "content_block_delta":
                yield {"type": "text", "content": chunk.delta.text}

    async def _stream_grok(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Grok (uses OpenAI format)"""
        client = self.ai_orchestrator.providers["grok"].client

        params = {
            "model": "grok-4",
            "messages": messages,
            "stream": True,
            "temperature": 0.8,  # Grok is more creative
        }

        if tools:
            params["tools"] = tools

        stream = await asyncio.to_thread(client.chat.completions.create, **params)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "text", "content": chunk.choices[0].delta.content}

    async def _extract_entities(self, query: str, context: GoldenEyeContext) -> Dict[str, Any]:
        """Extract entities from query"""
        # Simple entity extraction - in production, use NER or LLM
        entities = {"symbol": context.symbol, "timeframe": context.timeframe}

        # Extract symbol if mentioned
        import re

        symbol_match = re.search(r"\b[A-Z]{1,5}\b", query)
        if symbol_match:
            entities["symbol"] = symbol_match.group()

        # Extract time horizon
        horizon_match = re.search(r"(\d+)\s*(hour|day|week)", query.lower())
        if horizon_match:
            num = int(horizon_match.group(1))
            unit = horizon_match.group(2)
            if unit == "hour":
                entities["horizon"] = num
            elif unit == "day":
                entities["horizon"] = num * 24
            elif unit == "week":
                entities["horizon"] = num * 24 * 7

        # Extract agent names
        agent_names = ["RSI", "MACD", "Pattern", "Volume", "Sentiment", "Options"]
        mentioned_agents = []
        for agent in agent_names:
            if agent.lower() in query.lower():
                mentioned_agents.append(f"{agent}Agent")
        if mentioned_agents:
            entities["agents"] = mentioned_agents

        return entities

    def _get_intent_handler(self, intent: QueryIntent):
        """Get the appropriate handler for the intent"""
        handlers = {
            QueryIntent.PREDICTION: self._handle_prediction_query,
            QueryIntent.TECHNICAL_ANALYSIS: self._handle_technical_analysis_query,
            QueryIntent.RISK_ASSESSMENT: self._handle_risk_assessment_query,
            QueryIntent.MARKET_SENTIMENT: self._handle_sentiment_query,
            QueryIntent.ENTRY_EXIT_SIGNAL: self._handle_signal_query,
            QueryIntent.WORKFLOW_EXECUTION: self._handle_workflow_execution,
            QueryIntent.LIVE_CHART_ACTION: self._handle_chart_action_query,
            QueryIntent.GENERAL_QUESTION: self._handle_general_query,
        }
        return handlers.get(intent, self._handle_general_query)

    async def _handle_risk_assessment_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle risk assessment queries"""
        # Use Claude for risk assessment
        yield {"type": "thinking", "message": "Assessing risk factors..."}

        # Execute risk workflow
        risk_result = await execute_agent_workflow(
            workflow_name="risk_assessment", symbol=context.symbol
        )

        yield {"type": "tool_execution", "tool": "risk_assessment_workflow", "result": risk_result}

        # Claude interprets the risk
        messages = [
            {
                "role": "system",
                "content": "You are Claude, specialized in comprehensive risk assessment. Analyze these risk factors.",
            },
            {
                "role": "user",
                "content": f"Risk assessment for {context.symbol}: {json.dumps(risk_result, indent=2)}\n\nProvide risk score, key risks, and mitigation strategies.",
            },
        ]

        async for chunk in self._stream_claude(messages):
            yield chunk

    async def _handle_sentiment_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle sentiment analysis queries"""
        # Use Grok for real-time sentiment
        yield {"type": "thinking", "message": "Analyzing market sentiment..."}

        sentiment_result = await analyze_with_agent(
            "SentimentAgent", context.symbol, context.timeframe
        )

        yield {"type": "tool_execution", "tool": "sentiment_analysis", "result": sentiment_result}

        # Grok provides real-time context
        messages = [
            {
                "role": "system",
                "content": "You are Grok, with access to real-time information. Analyze current market sentiment.",
            },
            {
                "role": "user",
                "content": f"Sentiment analysis for {context.symbol}: {json.dumps(sentiment_result, indent=2)}\n\nWhat's the current market mood and recent news?",
            },
        ]

        async for chunk in self._stream_grok(messages):
            yield chunk

    async def _handle_signal_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle entry/exit signal queries"""
        signal_type = "entry" if "entry" in query.lower() else "exit"

        yield {"type": "thinking", "message": f"Calculating {signal_type} signals..."}

        # Execute appropriate workflow
        workflow_name = f"{signal_type}_signal"
        signal_result = await execute_agent_workflow(
            workflow_name=workflow_name, symbol=context.symbol
        )

        yield {
            "type": "tool_execution",
            "tool": f"{signal_type}_signal_workflow",
            "result": signal_result,
        }

        # Generate actionable recommendations
        async for chunk in self._stream_llm_response(
            llm_config["llm"],
            [
                {
                    "role": "system",
                    "content": f"Provide clear {signal_type} recommendations based on these signals.",
                },
                {
                    "role": "user",
                    "content": f"{signal_type.capitalize()} signal analysis: {json.dumps(signal_result, indent=2)}",
                },
            ],
        ):
            yield chunk

        # Add to chart
        if signal_result.get("success") and context.chart_visible:
            yield {
                "type": "chart_action",
                "action": {
                    "type": f"mark_{signal_type}_point",
                    "data": signal_result["recommendation"],
                },
            }

    async def _handle_chart_action_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle queries that require chart actions"""
        yield {"type": "thinking", "message": "Preparing chart visualization..."}

        # Determine what to draw
        if "support" in query.lower() or "resistance" in query.lower():
            # Calculate support/resistance levels
            pattern_result = await analyze_with_agent(
                "PatternAgent", context.symbol, context.timeframe
            )

            yield {
                "type": "chart_action",
                "action": {
                    "type": "draw_levels",
                    "levels": pattern_result.get("analysis", {}).get("key_levels", []),
                },
            }

        elif "pattern" in query.lower():
            # Detect and draw patterns
            pattern_result = await analyze_with_agent(
                "PatternAgent", context.symbol, context.timeframe
            )

            yield {
                "type": "chart_action",
                "action": {
                    "type": "highlight_pattern",
                    "patterns": pattern_result.get("analysis", {}).get("patterns", []),
                },
            }

        # Explain what was drawn
        yield {
            "type": "text",
            "content": "I've updated the chart with your requested visualization.",
        }

    async def _handle_general_query(
        self,
        query: str,
        entities: Dict[str, Any],
        llm_config: Dict[str, Any],
        context: GoldenEyeContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle general queries"""
        # Use the most appropriate LLM based on query
        messages = [
            {
                "role": "system",
                "content": "You are a helpful trading assistant with access to multiple analysis tools.",
            },
            {"role": "user", "content": query},
        ]

        # Provide tool access for general queries
        tools = (
            get_mcp_tools_for_openai()
            if llm_config["llm"] != "anthropic"
            else get_mcp_tools_for_anthropic()
        )

        async for chunk in self._stream_llm_response(llm_config["llm"], messages, tools):
            yield chunk


class IntentClassifier:
    """Classify user query intent"""

    async def classify(self, query: str) -> QueryIntent:
        """Classify query into intent categories"""
        query_lower = query.lower()

        # Simple keyword-based classification
        # In production, use a trained classifier or LLM

        if any(
            word in query_lower for word in ["predict", "forecast", "will", "future", "tomorrow"]
        ):
            return QueryIntent.PREDICTION

        elif any(
            word in query_lower for word in ["technical", "rsi", "macd", "indicator", "analysis"]
        ):
            return QueryIntent.TECHNICAL_ANALYSIS

        elif any(word in query_lower for word in ["risk", "safety", "exposure", "downside"]):
            return QueryIntent.RISK_ASSESSMENT

        elif any(word in query_lower for word in ["sentiment", "news", "social", "mood"]):
            return QueryIntent.MARKET_SENTIMENT

        elif any(word in query_lower for word in ["entry", "exit", "buy", "sell", "signal"]):
            return QueryIntent.ENTRY_EXIT_SIGNAL

        elif any(word in query_lower for word in ["workflow", "complete analysis", "full report"]):
            return QueryIntent.WORKFLOW_EXECUTION

        elif any(word in query_lower for word in ["draw", "show", "highlight", "chart"]):
            return QueryIntent.LIVE_CHART_ACTION

        else:
            return QueryIntent.GENERAL_QUESTION


class LLMRouter:
    """Route queries to appropriate LLM based on specialization"""

    def route(self, intent: QueryIntent, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which LLM and agents to use"""

        routing_map = {
            QueryIntent.PREDICTION: {
                "llm": "openai",
                "llm_name": "GPT-4o",
                "specialty": "technical analysis and predictions",
                "agents": ["LSTMForecastAgent", "PatternAgent", "MomentumAgent"],
                "tools": get_mcp_tools_for_openai(),
            },
            QueryIntent.TECHNICAL_ANALYSIS: {
                "llm": "openai",
                "llm_name": "GPT-4o",
                "specialty": "technical indicators and patterns",
                "agents": ["RSIAgent", "MACDAgent", "PatternAgent", "VolumeAgent"],
                "tools": get_mcp_tools_for_openai(),
            },
            QueryIntent.RISK_ASSESSMENT: {
                "llm": "anthropic",
                "llm_name": "Claude 3 Opus",
                "specialty": "comprehensive risk analysis",
                "agents": ["OptionsChainAgent", "MarketRegimeAgent", "VolumeAgent"],
                "tools": get_mcp_tools_for_anthropic(),
            },
            QueryIntent.MARKET_SENTIMENT: {
                "llm": "grok",
                "llm_name": "Grok 4",
                "specialty": "real-time market sentiment and news",
                "agents": ["SentimentAgent", "MarketRegimeAgent"],
                "tools": get_mcp_tools_for_openai(),  # Grok uses OpenAI format
            },
            QueryIntent.ENTRY_EXIT_SIGNAL: {
                "llm": "openai",
                "llm_name": "GPT-4o",
                "specialty": "trading signals and timing",
                "agents": ["RSIAgent", "MACDAgent", "MomentumAgent", "VolumeAgent"],
                "tools": get_mcp_tools_for_openai(),
            },
            QueryIntent.WORKFLOW_EXECUTION: {
                "llm": "anthropic",
                "llm_name": "Claude 3 Opus",
                "specialty": "complex analysis synthesis",
                "agents": ["all"],  # Will use workflow-specific agents
                "tools": get_mcp_tools_for_anthropic(),
            },
        }

        # Default configuration
        default_config = {
            "llm": "openai",
            "llm_name": "GPT-4o",
            "specialty": "general trading analysis",
            "agents": ["RSIAgent", "MACDAgent", "PatternAgent"],
            "tools": get_mcp_tools_for_openai(),
        }

        config = routing_map.get(intent, default_config)

        # Override with specific agents if mentioned
        if entities.get("agents"):
            config["agents"] = entities["agents"]

        return config


class ExecutionSandbox:
    """Safe execution environment for LLM-generated code"""

    async def execute_code(
        self, code: str, context: Dict[str, Any], timeout: int = 5
    ) -> Dict[str, Any]:
        """Execute code in a sandboxed environment"""
        # This is a placeholder - in production, use Docker or similar
        return {"success": False, "error": "Code execution not yet implemented"}


class ToolDiscovery:
    """Help LLMs discover available MCP tools"""

    def get_tools_for_intent(self, intent: QueryIntent) -> List[str]:
        """Get relevant tools for a given intent"""
        intent_tools = {
            QueryIntent.PREDICTION: [
                "predict_with_ai",
                "analyze_with_lstmforecastagent",
                "analyze_with_patternagent",
            ],
            QueryIntent.TECHNICAL_ANALYSIS: [
                "get_agent_consensus",
                "analyze_with_rsiagent",
                "analyze_with_macdagent",
                "analyze_with_volumeagent",
            ],
            QueryIntent.RISK_ASSESSMENT: [
                "execute_agent_workflow",
                "analyze_with_optionschainagent",
                "analyze_with_marketregimeagent",
            ],
        }

        return intent_tools.get(intent, ["get_agent_consensus"])


# Export the orchestrator
golden_eye_orchestrator = GoldenEyeOrchestrator()

__all__ = ["GoldenEyeOrchestrator", "GoldenEyeContext", "QueryIntent", "golden_eye_orchestrator"]
