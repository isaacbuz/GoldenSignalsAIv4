"""
GoldenSignalsAI Agent Bridge MCP Server - Week 3 Implementation
Bridges actual trading agents to MCP for real-time signal generation
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp import types

# Import the orchestrator and agents
from agents.orchestration.simple_orchestrator import SimpleOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentBridgeMCP(Server):
    """MCP server that bridges real trading agents"""

    def __init__(self):
        super().__init__("goldensignals-agent-bridge")
        logger.info("Initializing AgentBridgeMCP server...")

        # Initialize the orchestrator with default symbols
        self.orchestrator = SimpleOrchestrator(
            symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'META', 'AMZN']
        )

        # Start background signal generation (every 5 minutes)
        self.orchestrator.start_signal_generation(interval_seconds=300)

        # Track active agents
        self.active_agents = list(self.orchestrator.agents.keys())

        logger.info(f"AgentBridgeMCP server initialized with {len(self.active_agents)} agents")

    async def handle_initialize(self) -> types.InitializeResult:
        """Initialize the MCP server with capabilities"""
        logger.info("Handling MCP initialization...")

        return types.InitializeResult(
            protocol_version="2024-11-05",
            capabilities=types.ServerCapabilities(
                tools=types.ToolsCapability(list_changed=False),
                resources=types.ResourcesCapability(subscribe=True, list_changed=True)
            ),
            server_info=types.Implementation(
                name="GoldenSignals Agent Bridge Server",
                version="3.0.0"
            )
        )

    async def handle_list_tools(self) -> List[types.Tool]:
        """List available agent bridge tools"""
        logger.info("Listing available MCP tools...")

        return [
            types.Tool(
                name="generate_signal",
                description="Generate trading signal for a symbol using all active agents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, TSLA)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            types.Tool(
                name="get_agent_breakdown",
                description="Get detailed breakdown of how each agent voted for a symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            types.Tool(
                name="list_active_agents",
                description="List all active trading agents and their types",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="get_agent_performance",
                description="Get performance metrics for all agents",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="run_specific_agent",
                description="Run a specific agent on a symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Name of the agent (e.g., rsi, macd, bollinger)"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        }
                    },
                    "required": ["agent_name", "symbol"]
                }
            ),
            types.Tool(
                name="get_batch_signals",
                description="Generate signals for multiple symbols at once",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock symbols"
                        }
                    },
                    "required": ["symbols"]
                }
            ),
            types.Tool(
                name="get_signal_history",
                description="Get recent signal history for a symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of recent signals to retrieve",
                            "default": 5
                        }
                    },
                    "required": ["symbol"]
                }
            )
        ]

    async def _generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal for a symbol"""
        try:
            # Run signal generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            signal = await loop.run_in_executor(
                None,
                self.orchestrator.generate_signals_for_symbol,
                symbol.upper()
            )

            # Transform to user-friendly format
            return {
                "symbol": signal['symbol'],
                "action": signal['action'],
                "confidence": round(signal['confidence'], 3),
                "strength": self._get_strength(signal['confidence']),
                "reasoning": signal['metadata']['reasoning'],
                "timestamp": signal['timestamp'],
                "consensus": {
                    "total_agents": signal['metadata']['total_agents'],
                    "breakdown": self._summarize_breakdown(signal['metadata']['agent_breakdown'])
                }
            }

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            raise

    def _get_strength(self, confidence: float) -> str:
        """Convert confidence to strength label"""
        if confidence >= 0.8:
            return "STRONG"
        elif confidence >= 0.6:
            return "MODERATE"
        else:
            return "WEAK"

    def _summarize_breakdown(self, breakdown: Dict[str, Any]) -> Dict[str, int]:
        """Summarize agent votes"""
        summary = {"BUY": 0, "SELL": 0, "HOLD": 0, "ERROR": 0}

        for agent, data in breakdown.items():
            action = data.get('action', 'ERROR')
            if action in summary:
                summary[action] += 1
            else:
                summary['ERROR'] += 1

        return summary

    async def _get_agent_breakdown(self, symbol: str) -> Dict[str, Any]:
        """Get detailed agent breakdown"""
        try:
            # Get latest signal or generate new one
            signal = self.orchestrator.get_latest_signal(symbol.upper())
            if not signal:
                signal = await self._generate_signal(symbol)

            breakdown = signal.get('metadata', {}).get('agent_breakdown', {})

            # Organize by agent category
            categorized = {
                "technical": {},
                "volume": {},
                "market": {},
                "sentiment": {},
                "options": {},
                "advanced": {}
            }

            # Categorize agents
            for agent_name, data in breakdown.items():
                if agent_name in ['rsi', 'macd', 'ma_cross', 'bollinger', 'stochastic', 'ema', 'atr', 'vwap']:
                    categorized['technical'][agent_name] = data
                elif agent_name in ['volume', 'volume_profile']:
                    categorized['volume'][agent_name] = data
                elif agent_name in ['market_profile', 'order_flow']:
                    categorized['market'][agent_name] = data
                elif agent_name == 'sentiment':
                    categorized['sentiment'][agent_name] = data
                elif agent_name == 'options_flow':
                    categorized['options'][agent_name] = data
                else:
                    categorized['advanced'][agent_name] = data

            return {
                "symbol": symbol.upper(),
                "consensus": {
                    "action": signal.get('action', 'HOLD'),
                    "confidence": signal.get('confidence', 0),
                    "reasoning": signal.get('metadata', {}).get('reasoning', '')
                },
                "agent_categories": categorized,
                "summary": self._summarize_breakdown(breakdown),
                "timestamp": signal.get('timestamp', datetime.now().isoformat())
            }

        except Exception as e:
            logger.error(f"Error getting agent breakdown for {symbol}: {e}")
            raise

    async def _list_active_agents(self) -> Dict[str, Any]:
        """List all active agents"""
        agents_by_phase = {
            "phase_1": ['rsi', 'macd', 'volume', 'ma_cross'],
            "phase_2": ['bollinger', 'stochastic', 'ema', 'atr', 'vwap'],
            "phase_3": ['ichimoku', 'fibonacci', 'adx', 'parabolic_sar', 'std_dev'],
            "phase_4": ['volume_profile', 'market_profile', 'order_flow', 'sentiment', 'options_flow']
        }

        agent_details = {}
        for agent_name in self.active_agents:
            agent = self.orchestrator.agents.get(agent_name)
            if agent:
                agent_details[agent_name] = {
                    "type": agent.__class__.__name__,
                    "active": True,
                    "phase": self._get_agent_phase(agent_name, agents_by_phase)
                }

        return {
            "total_agents": len(self.active_agents),
            "agents": agent_details,
            "phases": agents_by_phase,
            "orchestrator_status": "running" if self.orchestrator.running else "stopped"
        }

    def _get_agent_phase(self, agent_name: str, phases: Dict[str, List[str]]) -> str:
        """Get the phase of an agent"""
        for phase, agents in phases.items():
            if agent_name in agents:
                return phase
        return "unknown"

    async def _get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            self.orchestrator.get_performance_metrics
        )

        # Add additional insights
        top_performers = sorted(
            [(name, data['avg_confidence']) for name, data in metrics['agents'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        metrics['insights'] = {
            "top_performers": [{"agent": name, "avg_confidence": conf} for name, conf in top_performers],
            "most_active": sorted(
                [(name, data['total_signals']) for name, data in metrics['agents'].items()],
                key=lambda x: x[1],
                reverse=True
            )[0][0] if metrics['agents'] else None
        }

        return metrics

    async def _run_specific_agent(self, agent_name: str, symbol: str) -> Dict[str, Any]:
        """Run a specific agent"""
        try:
            agent = self.orchestrator.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")

            # Run agent
            loop = asyncio.get_event_loop()
            signal = await loop.run_in_executor(
                None,
                self.orchestrator.run_agent,
                agent_name,
                agent,
                symbol.upper()
            )

            return {
                "agent": agent_name,
                "symbol": symbol.upper(),
                "signal": {
                    "action": signal.get('action', 'HOLD'),
                    "confidence": signal.get('confidence', 0),
                    "reasoning": signal.get('metadata', {}).get('reasoning', ''),
                    "indicators": signal.get('metadata', {}).get('indicators', {})
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error running agent {agent_name} for {symbol}: {e}")
            raise

    async def _get_batch_signals(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate signals for multiple symbols"""
        tasks = []
        for symbol in symbols[:10]:  # Limit to 10 symbols
            tasks.append(self._generate_signal(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "symbol": symbols[i],
                    "error": str(result)
                })
            else:
                signals.append(result)

        return {
            "signals": signals,
            "errors": errors,
            "total_requested": len(symbols),
            "total_processed": len(signals),
            "timestamp": datetime.now().isoformat()
        }

    async def _get_signal_history(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Get signal history (mock implementation)"""
        # In a real implementation, this would query a database
        # For now, we'll generate a few signals with timestamps

        history = []
        current_signal = await self._generate_signal(symbol)
        history.append(current_signal)

        # Mock historical data
        actions = ['BUY', 'HOLD', 'SELL']
        for i in range(1, min(limit, 5)):
            mock_signal = {
                "symbol": symbol.upper(),
                "action": actions[i % 3],
                "confidence": 0.5 + (i * 0.1),
                "strength": "MODERATE",
                "reasoning": f"Historical signal {i}",
                "timestamp": datetime.now().isoformat()
            }
            history.append(mock_signal)

        return {
            "symbol": symbol.upper(),
            "history": history,
            "count": len(history),
            "latest": history[0] if history else None
        }

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute tool calls"""
        logger.info(f"Executing tool: {name} with arguments: {arguments}")

        try:
            if name == "generate_signal":
                result = await self._generate_signal(arguments["symbol"])

            elif name == "get_agent_breakdown":
                result = await self._get_agent_breakdown(arguments["symbol"])

            elif name == "list_active_agents":
                result = await self._list_active_agents()

            elif name == "get_agent_performance":
                result = await self._get_agent_performance()

            elif name == "run_specific_agent":
                result = await self._run_specific_agent(
                    arguments["agent_name"],
                    arguments["symbol"]
                )

            elif name == "get_batch_signals":
                result = await self._get_batch_signals(arguments["symbols"])

            elif name == "get_signal_history":
                result = await self._get_signal_history(
                    arguments["symbol"],
                    arguments.get("limit", 5)
                )

            else:
                raise ValueError(f"Unknown tool: {name}")

            # Return formatted result
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            error_result = {
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }
            return [types.TextContent(
                type="text",
                text=json.dumps(error_result, indent=2)
            )]

    async def handle_list_resources(self) -> List[types.Resource]:
        """List available resources for subscription"""
        logger.info("Listing available MCP resources...")

        return [
            types.Resource(
                uri="agents://signals/live",
                name="Live Trading Signals",
                description="Real-time trading signals from all agents",
                mime_type="application/json"
            ),
            types.Resource(
                uri="agents://performance/metrics",
                name="Agent Performance Metrics",
                description="Real-time performance metrics for all agents",
                mime_type="application/json"
            ),
            types.Resource(
                uri="agents://status/health",
                name="System Health Status",
                description="Health and status of all trading agents",
                mime_type="application/json"
            )
        ]

    async def handle_read_resource(self, uri: str) -> str:
        """Read resource data"""
        logger.info(f"Reading resource: {uri}")

        try:
            if uri == "agents://signals/live":
                # Get latest signals for tracked symbols
                signals = []
                for symbol in self.orchestrator.symbols[:5]:  # Top 5 symbols
                    try:
                        signal = await self._generate_signal(symbol)
                        signals.append(signal)
                    except:
                        continue

                result = {
                    "type": "live_signals",
                    "signals": signals,
                    "timestamp": datetime.now().isoformat()
                }

            elif uri == "agents://performance/metrics":
                result = await self._get_agent_performance()

            elif uri == "agents://status/health":
                # System health check
                agent_status = {}
                for agent_name, agent in self.orchestrator.agents.items():
                    agent_status[agent_name] = "healthy"  # Simplified

                result = {
                    "type": "system_health",
                    "orchestrator": "running" if self.orchestrator.running else "stopped",
                    "agents": agent_status,
                    "total_agents": len(self.orchestrator.agents),
                    "active_symbols": len(self.orchestrator.symbols),
                    "timestamp": datetime.now().isoformat()
                }

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return json.dumps({"error": str(e), "uri": uri})

    def __del__(self):
        """Cleanup when server stops"""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.stop()

# Run the server
if __name__ == "__main__":
    import mcp.server.stdio

    async def main():
        logger.info("Starting GoldenSignals Agent Bridge MCP Server...")

        try:
            server = AgentBridgeMCP()

            # Run the MCP server
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.handle_initialize,
                    server.handle_list_tools,
                    server.handle_call_tool,
                    server.handle_list_resources,
                    server.handle_read_resource
                )
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise

    # Run the async main function
    asyncio.run(main())
