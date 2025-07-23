"""
MCP (Model Context Protocol) Tools Implementation
Provides standardized tool interfaces for all AI providers
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools available through MCP"""

    MARKET_DATA = "market_data"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    OPTIONS_FLOW = "options_flow"
    NEWS_SEARCH = "news_search"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    RISK_CALCULATION = "risk_calculation"
    BACKTESTING = "backtesting"


@dataclass
class ToolParameter:
    """Parameter definition for MCP tools"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """Complete tool definition for MCP"""

    name: str
    description: str
    type: ToolType
    parameters: List[ToolParameter]
    returns: str
    examples: List[Dict[str, Any]] = None


@dataclass
class ToolResult:
    """Standardized result from tool execution"""

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


def mcp_tool(tool_type: ToolType, description: str):
    """Decorator to register a function as an MCP tool"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            try:
                # Execute the tool
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                execution_time = asyncio.get_event_loop().time() - start_time

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={
                        "tool_name": func.__name__,
                        "tool_type": tool_type.value,
                        "timestamp": datetime.now().isoformat(),
                    },
                    execution_time=execution_time,
                )
            except Exception as e:
                logger.error(f"Tool {func.__name__} failed: {e}")
                return ToolResult(
                    success=False,
                    data=None,
                    error=str(e),
                    metadata={
                        "tool_name": func.__name__,
                        "tool_type": tool_type.value,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        # Store metadata on the wrapper
        wrapper.tool_type = tool_type
        wrapper.description = description
        wrapper.is_mcp_tool = True

        return wrapper

    return decorator


class MCPToolRegistry:
    """Registry for all MCP tools"""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.definitions: Dict[str, ToolDefinition] = {}
        self._register_standard_tools()

    def register(self, tool_def: ToolDefinition, implementation: Callable):
        """Register a tool with its definition"""
        self.tools[tool_def.name] = implementation
        self.definitions[tool_def.name] = tool_def
        logger.info(f"Registered MCP tool: {tool_def.name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[ToolDefinition]:
        """List available tools, optionally filtered by type"""
        tools = list(self.definitions.values())
        if tool_type:
            tools = [t for t in tools if t.type == tool_type]
        return tools

    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format"""
        functions = []
        for tool_def in self.definitions.values():
            function = {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

            for param in tool_def.parameters:
                prop = {"type": param.type, "description": param.description}
                if param.enum:
                    prop["enum"] = param.enum

                function["parameters"]["properties"][param.name] = prop

                if param.required:
                    function["parameters"]["required"].append(param.name)

            functions.append(function)

        return functions

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic tool use format"""
        tools = []
        for tool_def in self.definitions.values():
            tool = {
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }

            for param in tool_def.parameters:
                prop = {"type": param.type, "description": param.description}
                if param.enum:
                    prop["enum"] = param.enum

                tool["input_schema"]["properties"][param.name] = prop

                if param.required:
                    tool["input_schema"]["required"].append(param.name)

            tools.append(tool)

        return tools

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool '{tool_name}' not found")

        try:
            return await tool(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def _register_standard_tools(self):
        """Register the standard set of trading tools"""

        # Market Data Tool
        @mcp_tool(ToolType.MARKET_DATA, "Fetch real-time market data for a symbol")
        async def get_market_data(symbol: str, period: str = "1d") -> Dict[str, Any]:
            """Get market data including price, volume, and key metrics"""
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period)

            return {
                "symbol": symbol,
                "current_price": info.get(
                    "regularMarketPrice", hist["Close"].iloc[-1] if not hist.empty else None
                ),
                "volume": info.get("volume", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "price_history": hist.to_dict() if not hist.empty else {},
            }

        # Technical Analysis Tool
        @mcp_tool(ToolType.TECHNICAL_ANALYSIS, "Calculate technical indicators")
        async def calculate_technicals(
            symbol: str, indicators: List[str], period: str = "30d"
        ) -> Dict[str, Any]:
            """Calculate various technical indicators"""
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            results = {"symbol": symbol}

            if "RSI" in indicators:
                # Simple RSI calculation
                delta = data["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                results["RSI"] = (100 - (100 / (1 + rs))).iloc[-1]

            if "SMA" in indicators:
                results["SMA_20"] = data["Close"].rolling(window=20).mean().iloc[-1]
                results["SMA_50"] = data["Close"].rolling(window=50).mean().iloc[-1]

            if "MACD" in indicators:
                exp1 = data["Close"].ewm(span=12, adjust=False).mean()
                exp2 = data["Close"].ewm(span=26, adjust=False).mean()
                results["MACD"] = (exp1 - exp2).iloc[-1]
                results["MACD_signal"] = (exp1 - exp2).ewm(span=9, adjust=False).mean().iloc[-1]

            return results

        # Sentiment Analysis Tool
        @mcp_tool(ToolType.SENTIMENT_ANALYSIS, "Analyze market sentiment")
        async def analyze_sentiment(
            symbol: str, sources: List[str] = ["news", "social"]
        ) -> Dict[str, Any]:
            """Analyze sentiment from various sources"""
            # This is a placeholder - in production, integrate with real sentiment APIs
            return {
                "symbol": symbol,
                "overall_sentiment": "bullish",
                "sentiment_score": 0.72,
                "sources_analyzed": sources,
                "key_topics": ["earnings beat", "product launch", "market expansion"],
                "confidence": 0.85,
            }

        # Options Flow Tool
        @mcp_tool(ToolType.OPTIONS_FLOW, "Analyze options flow data")
        async def analyze_options_flow(symbol: str, expiry: Optional[str] = None) -> Dict[str, Any]:
            """Analyze options chain and flow"""
            ticker = yf.Ticker(symbol)

            # Get options chain
            expirations = ticker.options
            if not expirations:
                return {"error": "No options data available"}

            # Use provided expiry or first available
            expiry_to_use = expiry if expiry in expirations else expirations[0]
            opt_chain = ticker.option_chain(expiry_to_use)

            # Calculate put/call ratio and other metrics
            calls = opt_chain.calls
            puts = opt_chain.puts

            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()
            pc_ratio = put_volume / call_volume if call_volume > 0 else 0

            return {
                "symbol": symbol,
                "expiry": expiry_to_use,
                "put_call_ratio": pc_ratio,
                "call_volume": int(call_volume),
                "put_volume": int(put_volume),
                "max_pain": float(calls["strike"].iloc[len(calls) // 2]),  # Simplified
                "unusual_activity": pc_ratio > 1.5 or pc_ratio < 0.5,
            }

        # Risk Calculator Tool
        @mcp_tool(ToolType.RISK_CALCULATION, "Calculate position risk metrics")
        async def calculate_risk(
            symbol: str,
            entry_price: float,
            stop_loss: float,
            position_size: float,
            portfolio_value: float,
        ) -> Dict[str, Any]:
            """Calculate comprehensive risk metrics"""
            # Calculate basic risk metrics
            risk_per_share = abs(entry_price - stop_loss)
            total_risk = risk_per_share * position_size
            risk_percentage = (total_risk / portfolio_value) * 100

            # Get historical volatility
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            returns = hist["Close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Kelly Criterion calculation (simplified)
            win_rate = 0.55  # Placeholder
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss
            kelly_percentage = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

            return {
                "symbol": symbol,
                "position_risk": total_risk,
                "portfolio_risk_pct": risk_percentage,
                "volatility_annual": volatility,
                "recommended_position_size": min(position_size, portfolio_value * kelly_percentage),
                "risk_reward_ratio": 2.0,  # Placeholder
                "max_drawdown_expected": volatility * 2,
            }

        # Register tool definitions
        self.register(
            ToolDefinition(
                name="get_market_data",
                description="Fetch real-time market data for a symbol",
                type=ToolType.MARKET_DATA,
                parameters=[
                    ToolParameter("symbol", "string", "Stock symbol", True),
                    ToolParameter(
                        "period",
                        "string",
                        "Time period",
                        False,
                        "1d",
                        ["1d", "5d", "1mo", "3mo", "1y"],
                    ),
                ],
                returns="Market data including price, volume, and metrics",
            ),
            get_market_data,
        )

        self.register(
            ToolDefinition(
                name="calculate_technicals",
                description="Calculate technical indicators for a symbol",
                type=ToolType.TECHNICAL_ANALYSIS,
                parameters=[
                    ToolParameter("symbol", "string", "Stock symbol", True),
                    ToolParameter("indicators", "array", "List of indicators", True),
                    ToolParameter("period", "string", "Time period", False, "30d"),
                ],
                returns="Technical indicator values",
            ),
            calculate_technicals,
        )

        self.register(
            ToolDefinition(
                name="analyze_sentiment",
                description="Analyze market sentiment for a symbol",
                type=ToolType.SENTIMENT_ANALYSIS,
                parameters=[
                    ToolParameter("symbol", "string", "Stock symbol", True),
                    ToolParameter("sources", "array", "Data sources", False, ["news", "social"]),
                ],
                returns="Sentiment analysis results",
            ),
            analyze_sentiment,
        )

        self.register(
            ToolDefinition(
                name="analyze_options_flow",
                description="Analyze options flow and unusual activity",
                type=ToolType.OPTIONS_FLOW,
                parameters=[
                    ToolParameter("symbol", "string", "Stock symbol", True),
                    ToolParameter("expiry", "string", "Option expiry date", False),
                ],
                returns="Options flow analysis",
            ),
            analyze_options_flow,
        )

        self.register(
            ToolDefinition(
                name="calculate_risk",
                description="Calculate position risk metrics",
                type=ToolType.RISK_CALCULATION,
                parameters=[
                    ToolParameter("symbol", "string", "Stock symbol", True),
                    ToolParameter("entry_price", "number", "Entry price", True),
                    ToolParameter("stop_loss", "number", "Stop loss price", True),
                    ToolParameter("position_size", "number", "Number of shares", True),
                    ToolParameter("portfolio_value", "number", "Total portfolio value", True),
                ],
                returns="Risk metrics and recommendations",
            ),
            calculate_risk,
        )


# Singleton instance
mcp_registry = MCPToolRegistry()


# Export convenience functions
def get_mcp_tools_for_openai() -> List[Dict[str, Any]]:
    """Get tools formatted for OpenAI function calling"""
    return mcp_registry.get_openai_functions()


def get_mcp_tools_for_anthropic() -> List[Dict[str, Any]]:
    """Get tools formatted for Anthropic tool use"""
    return mcp_registry.get_anthropic_tools()


async def execute_mcp_tool(tool_name: str, **kwargs) -> ToolResult:
    """Execute an MCP tool by name"""
    return await mcp_registry.execute(tool_name, **kwargs)
