"""
GoldenSignalsAI MCP Server - Week 1 Implementation
Wraps the existing SimpleOrchestrator to expose trading signals via MCP
"""

import asyncio
import json
import logging
from typing import Any, Dict, List
from mcp.server import Server
from mcp import types
import sys
import os

# Add parent directory to path to import orchestrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing orchestrator
from agents.orchestration.simple_orchestrator import SimpleOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSignalsMCP(Server):
    """MCP server that exposes GoldenSignalsAI trading signals"""

    def __init__(self):
        super().__init__("goldensignals-trading")
        logger.info("Initializing TradingSignalsMCP server...")

        # Initialize your existing orchestrator with default symbols
        self.orchestrator = SimpleOrchestrator(
            symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        )

        # Start the orchestrator's signal generation (optional)
        # self.orchestrator.start_signal_generation(interval_seconds=300)

        logger.info("TradingSignalsMCP server initialized successfully")

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
                name="GoldenSignals Trading Server",
                version="1.0.0"
            )
        )

    async def handle_list_tools(self) -> List[types.Tool]:
        """List available trading tools"""
        logger.info("Listing available MCP tools...")

        return [
            types.Tool(
                name="generate_signal",
                description="Generate trading signal for a symbol using all 19 agents with consensus",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, TSLA, SPY)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            types.Tool(
                name="get_agent_breakdown",
                description="Get detailed breakdown of all agent signals for a symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"}
                    },
                    "required": ["symbol"]
                }
            ),
            types.Tool(
                name="get_performance_metrics",
                description="Get performance metrics for all trading agents",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="get_all_signals",
                description="Generate signals for all tracked symbols",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute tool calls"""
        logger.info(f"Executing tool: {name} with arguments: {arguments}")

        try:
            if name == "generate_signal":
                symbol = arguments["symbol"].upper()

                # Use your existing orchestrator to generate signal
                signal = self.orchestrator.generate_signals_for_symbol(symbol)

                # Convert to frontend-friendly format
                result = self.orchestrator.to_json(signal)

                # Add summary for easier reading
                result["summary"] = (
                    f"Signal: {result['signal_type']} with "
                    f"{result['confidence']:.1%} confidence ({result['strength']})"
                )

            elif name == "get_agent_breakdown":
                symbol = arguments["symbol"].upper()

                # Generate signal to get full breakdown
                signal = self.orchestrator.generate_signals_for_symbol(symbol)

                # Extract agent breakdown with more details
                breakdown = signal["metadata"].get("agent_breakdown", {})

                # Count actions
                action_counts = {}
                for agent, data in breakdown.items():
                    action = data["action"]
                    action_counts[action] = action_counts.get(action, 0) + 1

                result = {
                    "symbol": symbol,
                    "consensus": {
                        "action": signal["action"],
                        "confidence": signal["confidence"],
                        "reasoning": signal["metadata"]["reasoning"]
                    },
                    "action_summary": action_counts,
                    "total_agents": len(breakdown),
                    "agents": breakdown
                }

            elif name == "get_performance_metrics":
                # Get performance metrics from orchestrator
                result = self.orchestrator.get_performance_metrics()

            elif name == "get_all_signals":
                # Generate signals for all symbols
                all_signals = self.orchestrator.generate_all_signals()

                # Convert each signal to JSON format
                result = {
                    "signals": [
                        self.orchestrator.to_json(signal)
                        for signal in all_signals
                    ],
                    "total_symbols": len(all_signals),
                    "timestamp": all_signals[0]["timestamp"] if all_signals else None
                }

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
                uri="signals://realtime",
                name="Real-time Trading Signals",
                description="Subscribe to real-time trading signals for all symbols",
                mime_type="application/json"
            ),
            types.Resource(
                uri="signals://agent-insights",
                name="Agent Insights Stream",
                description="Real-time insights from all 19 trading agents",
                mime_type="application/json"
            ),
            types.Resource(
                uri="signals://performance",
                name="Performance Metrics",
                description="Real-time performance metrics for all agents",
                mime_type="application/json"
            )
        ]

    async def handle_read_resource(self, uri: str) -> str:
        """Read resource data"""
        logger.info(f"Reading resource: {uri}")

        try:
            if uri == "signals://realtime":
                # Get latest signals for all symbols
                signals = self.orchestrator.generate_all_signals()
                result = {
                    "type": "realtime_signals",
                    "signals": [
                        self.orchestrator.to_json(signal)
                        for signal in signals
                    ],
                    "timestamp": signals[0]["timestamp"] if signals else None
                }

            elif uri == "signals://agent-insights":
                # Get latest insights from data bus
                # For now, return a summary of agent activities
                result = {
                    "type": "agent_insights",
                    "agents": list(self.orchestrator.agents.keys()),
                    "total_agents": len(self.orchestrator.agents),
                    "message": "Agent insights will be available when data bus is integrated"
                }

            elif uri == "signals://performance":
                # Get performance metrics
                result = self.orchestrator.get_performance_metrics()

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return json.dumps({"error": str(e), "uri": uri})

# Run the server
if __name__ == "__main__":
    import mcp.server.stdio

    async def main():
        logger.info("Starting GoldenSignals MCP Server...")

        try:
            server = TradingSignalsMCP()

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
