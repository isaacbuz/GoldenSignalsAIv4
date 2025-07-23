"""
AI Orchestrator Service
Intelligently routes requests to the best AI provider based on task requirements
Integrated with LangGraph for workflow management
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import openai
from anthropic import Anthropic
from langsmith import traceable

from services.langsmith_observability import trace_llm_call, trace_workflow
from services.redis_cache_service import redis_cache

logger = logging.getLogger(__name__)


class TaskType(Enum):
    MARKET_ANALYSIS = "market_analysis"
    REAL_TIME_NEWS = "real_time_news"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_REASONING = "trade_reasoning"
    CODE_GENERATION = "code_generation"
    LONG_CONTEXT = "long_context"


@dataclass
class AIProvider:
    name: str
    client: Any
    strengths: List[TaskType]
    cost_per_1k_tokens: float
    max_tokens: int
    supports_tools: bool
    supports_vision: bool


class AIOrchestrator:
    def __init__(self):
        self.providers = self._initialize_providers()

    def _initialize_providers(self) -> Dict[str, AIProvider]:
        providers = {}

        # OpenAI GPT-4o
        if os.getenv("OPENAI_API_KEY"):
            providers["openai"] = AIProvider(
                name="OpenAI GPT-4o",
                client=openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                strengths=[
                    TaskType.PATTERN_RECOGNITION,
                    TaskType.CODE_GENERATION,
                    TaskType.TRADE_REASONING,
                ],
                cost_per_1k_tokens=0.0025,  # Input cost
                max_tokens=128000,
                supports_tools=True,
                supports_vision=True,
            )

        # Anthropic Claude
        if os.getenv("ANTHROPIC_API_KEY"):
            providers["anthropic"] = AIProvider(
                name="Claude 3 Opus",
                client=Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
                strengths=[
                    TaskType.LONG_CONTEXT,
                    TaskType.RISK_ASSESSMENT,
                    TaskType.TRADE_REASONING,
                ],
                cost_per_1k_tokens=0.015,  # Input cost
                max_tokens=200000,
                supports_tools=True,
                supports_vision=True,
            )

        # xAI Grok 4
        if os.getenv("XAI_API_KEY"):
            providers["grok"] = AIProvider(
                name="Grok 4",
                client=openai.OpenAI(
                    api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1"
                ),
                strengths=[
                    TaskType.REAL_TIME_NEWS,
                    TaskType.MARKET_ANALYSIS,
                    TaskType.TRADE_REASONING,
                ],
                cost_per_1k_tokens=0.038,  # $5/131k tokens
                max_tokens=131072,
                supports_tools=True,
                supports_vision=False,  # Coming soon
            )

        return providers

    async def analyze_market(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate multiple AI providers for comprehensive market analysis
        Now with MCP tool integration for standardized access
        """
        # Import MCP tools
        from services.mcp_tools import (
            execute_mcp_tool,
            get_mcp_tools_for_anthropic,
            get_mcp_tools_for_openai,
        )

        tasks = []

        # 1. Grok 4 - Real-time market news and sentiment with MCP tools
        if "grok" in self.providers:
            tasks.append(self._grok_realtime_analysis_mcp(symbol))

        # 2. OpenAI - Technical pattern recognition with MCP tools
        if "openai" in self.providers:
            tasks.append(self._openai_pattern_analysis_mcp(symbol, context))

        # 3. Claude - Risk assessment and long-term analysis with MCP tools
        if "anthropic" in self.providers:
            tasks.append(self._claude_risk_analysis_mcp(symbol, context))

        # Run all analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine insights
        return self._combine_analyses(results)

    async def _grok_realtime_analysis(self, symbol: str) -> Dict[str, Any]:
        """Use Grok 4 for real-time market news and sentiment"""
        try:
            client = self.providers["grok"].client

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="grok-2-1212",  # Updated model name
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst with access to real-time market data. Use your live search capabilities to find the latest news and sentiment.",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the latest news and market sentiment for {symbol}. Use live search to find recent developments, analyst opinions, and market moving events from the last 24 hours.",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search for real-time information",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                    }
                ],
                temperature=0.3,
            )

            return {
                "provider": "grok",
                "analysis_type": "real_time_news",
                "content": response.choices[0].message.content,
                "confidence": 0.85,
            }

        except Exception as e:
            logger.error(f"Grok analysis failed: {e}")
            return {"provider": "grok", "error": str(e)}

    async def _openai_pattern_analysis(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use OpenAI for technical pattern recognition"""
        try:
            client = self.providers["openai"].client

            # Prepare price data for analysis
            price_data = context.get("price_history", [])

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical analyst specializing in pattern recognition and price action analysis.",
                    },
                    {
                        "role": "user",
                        "content": f"""
                    Analyze the following price data for {symbol} and identify:
                    1. Key chart patterns (head & shoulders, triangles, flags, etc.)
                    2. Support and resistance levels
                    3. Trend strength and direction
                    4. Volume patterns

                    Price data: {price_data[-100:]}  # Last 100 candles
                    """,
                    },
                ],
                temperature=0.2,
            )

            return {
                "provider": "openai",
                "analysis_type": "technical_patterns",
                "content": response.choices[0].message.content,
                "confidence": 0.80,
            }

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return {"provider": "openai", "error": str(e)}

    async def _claude_risk_analysis(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use Claude for comprehensive risk assessment"""
        try:
            client = self.providers["anthropic"].client

            # Include extensive context for Claude's analysis
            market_context = {
                "symbol": symbol,
                "historical_volatility": context.get("volatility", {}),
                "correlation_matrix": context.get("correlations", {}),
                "macro_indicators": context.get("macro", {}),
                "earnings_history": context.get("earnings", []),
                "sector_performance": context.get("sector", {}),
            }

            response = await asyncio.to_thread(
                client.messages.create,
                model="claude-3-opus-20240229",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                    Perform a comprehensive risk assessment for {symbol} considering:

                    1. Market Risk Factors:
                       - Historical volatility patterns
                       - Correlation with market indices
                       - Sector-specific risks

                    2. Company-Specific Risks:
                       - Earnings volatility
                       - Management changes
                       - Competitive positioning

                    3. Macro Risk Factors:
                       - Interest rate sensitivity
                       - Currency exposure
                       - Regulatory risks

                    Context: {market_context}

                    Provide a detailed risk score (1-10) with justification and specific risk mitigation strategies.
                    """,
                    }
                ],
                max_tokens=4000,
                temperature=0.3,
            )

            return {
                "provider": "anthropic",
                "analysis_type": "risk_assessment",
                "content": response.content[0].text,
                "confidence": 0.90,
            }

        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return {"provider": "anthropic", "error": str(e)}

    def _combine_analyses(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine insights from multiple AI providers"""
        combined = {
            "timestamp": asyncio.get_event_loop().time(),
            "analyses": {},
            "consensus": {},
            "recommendations": [],
        }

        for result in results:
            if "error" not in result:
                combined["analyses"][result["provider"]] = {
                    "type": result["analysis_type"],
                    "content": result["content"],
                    "confidence": result["confidence"],
                }

        # Generate consensus recommendations
        if len(combined["analyses"]) >= 2:
            combined["consensus"] = self._generate_consensus(combined["analyses"])

        return combined

    def _generate_consensus(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consensus from multiple AI analyses"""
        # This is a simplified consensus mechanism
        # In production, you'd want more sophisticated voting/weighting

        sentiments = []
        risk_scores = []

        for provider, analysis in analyses.items():
            content = analysis["content"].lower()

            # Extract sentiment (simplified)
            if "bullish" in content or "buy" in content:
                sentiments.append(1)
            elif "bearish" in content or "sell" in content:
                sentiments.append(-1)
            else:
                sentiments.append(0)

            # Extract risk scores if mentioned
            # (Would need more sophisticated parsing in production)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        return {
            "sentiment": "bullish"
            if avg_sentiment > 0.3
            else "bearish"
            if avg_sentiment < -0.3
            else "neutral",
            "confidence": min(
                0.95, sum(a["confidence"] for a in analyses.values()) / len(analyses)
            ),
            "providers_agree": len(set(sentiments)) == 1,
        }

    async def get_trade_recommendation(
        self, symbol: str, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get a final trade recommendation using the most suitable AI"""
        # Use Grok 4 for final recommendation due to its superior reasoning
        if "grok" in self.providers:
            provider = self.providers["grok"]
        elif "anthropic" in self.providers:
            provider = self.providers["anthropic"]
        else:
            provider = self.providers.get("openai")

        if not provider:
            return {"error": "No AI provider available"}

        # Prepare comprehensive prompt
        prompt = f"""
        Based on the following multi-AI analysis for {symbol}, provide a clear trading recommendation:

        Analyses:
        {analysis_results}

        Provide:
        1. Clear BUY/SELL/HOLD recommendation
        2. Confidence level (0-100%)
        3. Entry price target
        4. Stop loss level
        5. Take profit targets (3 levels)
        6. Position sizing recommendation
        7. Time horizon
        8. Key risks to monitor
        """

        # Make the API call based on provider type
        # (Implementation details omitted for brevity)

        return {
            "recommendation": "BUY",  # Example
            "confidence": 0.75,
            "targets": {"entry": 100, "stop_loss": 95, "take_profit": [105, 110, 120]},
        }

    @trace_llm_call("xai", "grok-4")
    async def _grok_realtime_analysis_mcp(self, symbol: str) -> Dict[str, Any]:
        """Use Grok 4 with MCP tools for real-time analysis"""
        try:
            # Check cache first
            cached_result = await redis_cache.get_ai_prediction(
                symbol, "grok", {"type": "realtime"}
            )
            if cached_result:
                logger.info(f"Using cached Grok analysis for {symbol}")
                return cached_result
            from services.mcp_tools import execute_mcp_tool, get_mcp_tools_for_openai

            client = self.providers["grok"].client

            # First, get market data using MCP tool
            market_data = await execute_mcp_tool("get_market_data", symbol=symbol)
            sentiment_data = await execute_mcp_tool("analyze_sentiment", symbol=symbol)

            # Prepare messages with tool results
            messages = [
                {
                    "role": "system",
                    "content": "You are Grok 4, specialized in real-time market analysis. Use the provided tools and data to analyze the market.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze {symbol} with focus on real-time market dynamics.

                    Market Data: {json.dumps(market_data.data, indent=2)}
                    Sentiment Data: {json.dumps(sentiment_data.data, indent=2)}

                    Provide insights on:
                    1. Current market momentum
                    2. Real-time news impact
                    3. Social sentiment trends
                    4. Short-term price action prediction""",
                },
            ]

            # Get response with function calling
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="grok-4",
                messages=messages,
                tools=get_mcp_tools_for_openai(),
                temperature=0.7,
                max_tokens=2000,
            )

            result = {
                "provider": "grok",
                "analysis_type": "real_time",
                "content": response.choices[0].message.content,
                "confidence": 0.85,
                "tools_used": ["get_market_data", "analyze_sentiment"],
            }

            # Cache the result
            await redis_cache.set_ai_prediction(
                symbol, "grok", result, ttl=60, context={"type": "realtime"}
            )

            return result

        except Exception as e:
            logger.error(f"Grok MCP analysis failed: {e}")
            return {"provider": "grok", "error": str(e)}

    @trace_llm_call("openai", "gpt-4o")
    async def _openai_pattern_analysis_mcp(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use OpenAI with MCP tools for pattern recognition"""
        try:
            from services.mcp_tools import execute_mcp_tool, get_mcp_tools_for_openai

            client = self.providers["openai"].client

            # Get technical indicators using MCP
            technicals = await execute_mcp_tool(
                "calculate_technicals", symbol=symbol, indicators=["RSI", "MACD", "SMA"]
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are GPT-4o specialized in technical pattern recognition. Analyze charts and identify trading patterns.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze technical patterns for {symbol}.

                    Technical Indicators: {json.dumps(technicals.data, indent=2)}

                    Identify:
                    1. Chart patterns (head & shoulders, triangles, etc.)
                    2. Support and resistance levels
                    3. Trend strength and direction
                    4. Key technical signals""",
                },
            ]

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=messages,
                tools=get_mcp_tools_for_openai(),
                temperature=0.3,
                max_tokens=1500,
            )

            # Check if tool calls were made
            if response.choices[0].message.tool_calls:
                # Execute any additional tool calls
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    await execute_mcp_tool(tool_name, **tool_args)

            return {
                "provider": "openai",
                "analysis_type": "pattern_recognition",
                "content": response.choices[0].message.content,
                "confidence": 0.88,
                "tools_used": ["calculate_technicals"],
            }

        except Exception as e:
            logger.error(f"OpenAI MCP analysis failed: {e}")
            return {"provider": "openai", "error": str(e)}

    @trace_llm_call("anthropic", "claude-3-opus")
    async def _claude_risk_analysis_mcp(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use Claude with MCP tools for comprehensive risk assessment"""
        try:
            from services.mcp_tools import execute_mcp_tool, get_mcp_tools_for_anthropic

            client = self.providers["anthropic"].client

            # Get multiple data points using MCP
            market_data = await execute_mcp_tool("get_market_data", symbol=symbol, period="3mo")
            options_flow = await execute_mcp_tool("analyze_options_flow", symbol=symbol)

            # Mock risk calculation (would use real entry/stop in production)
            current_price = market_data.data.get("current_price", 100)
            risk_metrics = await execute_mcp_tool(
                "calculate_risk",
                symbol=symbol,
                entry_price=current_price,
                stop_loss=current_price * 0.95,
                position_size=100,
                portfolio_value=100000,
            )

            response = await asyncio.to_thread(
                client.messages.create,
                model="claude-3-opus-20240229",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Perform comprehensive risk assessment for {symbol}.

                    Market Data: {json.dumps(market_data.data, indent=2)}
                    Options Flow: {json.dumps(options_flow.data, indent=2)}
                    Risk Metrics: {json.dumps(risk_metrics.data, indent=2)}

                    Provide detailed analysis of:
                    1. Market risk factors
                    2. Company-specific risks
                    3. Options flow implications
                    4. Position sizing recommendations
                    5. Risk mitigation strategies""",
                    }
                ],
                max_tokens=4000,
                temperature=0.3,
                tools=get_mcp_tools_for_anthropic(),
            )

            return {
                "provider": "anthropic",
                "analysis_type": "risk_assessment",
                "content": response.content[0].text,
                "confidence": 0.90,
                "tools_used": ["get_market_data", "analyze_options_flow", "calculate_risk"],
            }

        except Exception as e:
            logger.error(f"Claude MCP analysis failed: {e}")
            return {"provider": "anthropic", "error": str(e)}


# Singleton instance
ai_orchestrator = AIOrchestrator()


# Export for use in other modules
async def get_ai_analysis(symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for AI-powered analysis"""
    return await ai_orchestrator.analyze_market(symbol, context)


@traceable(name="langgraph_trading_decision")
async def execute_trading_workflow(
    symbol: str, price: float, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the LangGraph trading workflow with AI orchestration
    This integrates multi-LLM analysis with agent-based decision making
    """
    try:
        # Import the LangGraph workflow
        from src.workflows.trading_workflow_langgraph import run_trading_analysis

        # First, get AI analysis from multiple providers
        ai_context = context or {}
        ai_analysis = await ai_orchestrator.analyze_market(symbol, ai_context)

        # Execute the LangGraph workflow
        workflow_result = await run_trading_analysis(symbol, price)

        # Enhance the workflow result with AI insights
        if workflow_result.get("execute"):
            # Get final recommendation from AI
            recommendation = await ai_orchestrator.get_trade_recommendation(symbol, ai_analysis)

            # Merge AI recommendation with workflow decision
            workflow_result["ai_enhanced"] = {
                "multi_llm_consensus": ai_analysis.get("consensus", {}),
                "ai_confidence": recommendation.get("confidence", 0),
                "ai_risk_factors": ai_analysis.get("analyses", {})
                .get("anthropic", {})
                .get("content", ""),
                "real_time_context": ai_analysis.get("analyses", {})
                .get("grok", {})
                .get("content", ""),
            }

            # Adjust position size based on AI confidence
            ai_confidence = recommendation.get("confidence", 0.5)
            workflow_result["position_size"] *= ai_confidence

        return workflow_result

    except Exception as e:
        logger.error(f"Trading workflow execution failed: {e}")
        return {"execute": False, "action": "HOLD", "error": str(e)}


def visualize_trading_workflow() -> str:
    """Get a visual representation of the trading workflow"""
    try:
        from src.workflows.trading_workflow_langgraph import get_workflow_visualization

        return get_workflow_visualization()
    except Exception as e:
        logger.error(f"Failed to get workflow visualization: {e}")
        return "Unable to generate visualization"
