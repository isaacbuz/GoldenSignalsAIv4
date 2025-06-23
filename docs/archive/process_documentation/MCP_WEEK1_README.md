# GoldenSignalsAI MCP Server - Week 1 Implementation

## 🚀 Quick Start

Week 1 implementation is complete! You now have a working MCP server that exposes your trading signals to Claude Desktop.

## 📁 What We Created

1. **`mcp_servers/simple_trading_mcp.py`** - A simplified MCP server that:
   - Exposes trading signal generation tools
   - Simulates 19 different trading agents
   - Provides consensus-based signals
   - Works without complex dependencies

2. **`claude_desktop_config.json`** - Configuration file for Claude Desktop
   - Already copied to `~/Library/Application Support/Claude/`
   - Points to your MCP server

## 🛠️ Available MCP Tools

Your MCP server exposes three main tools:

### 1. `generate_signal`
Generate a trading signal for any stock symbol using consensus from 19 simulated agents.

**Example usage in Claude:**
```
Generate a trading signal for AAPL
```

### 2. `get_agent_breakdown`
Get a detailed breakdown showing how each of the 19 agents voted.

**Example usage in Claude:**
```
Show me the agent breakdown for TSLA
```

### 3. `get_all_signals`
Generate signals for all tracked symbols (AAPL, GOOGL, MSFT, TSLA, NVDA, SPY, QQQ).

**Example usage in Claude:**
```
Generate signals for all tracked stocks
```

## 📊 Available Resources

The server also provides subscribable resources:

1. **`signals://realtime`** - Real-time trading signals
2. **`signals://performance`** - Performance metrics for all agents

## 🔧 Testing Your MCP Server

### Step 1: Restart Claude Desktop
After the configuration has been copied, restart Claude Desktop to load the MCP server.

### Step 2: Test Basic Commands
Try these commands in Claude:

1. "Generate a trading signal for SPY"
2. "Show me all agent opinions for AAPL"
3. "Generate signals for all stocks"

### Step 3: Check the Logs
If something doesn't work, check the MCP logs:
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

## 🐛 Troubleshooting

### Issue: Claude doesn't recognize the commands
**Solution:** Make sure Claude Desktop has been restarted after copying the config.

### Issue: Python not found
**Solution:** Update the config to use the full Python path:
```json
{
  "mcpServers": {
    "goldensignals": {
      "command": "/Users/isaacbuz/Documents/Projects/FinTech/GoldenSignalsAI_V2/.venv/bin/python",
      "args": ["/Users/isaacbuz/Documents/Projects/FinTech/GoldenSignalsAI_V2/mcp_servers/simple_trading_mcp.py"]
    }
  }
}
```

### Issue: Server crashes
**Solution:** Test the server manually:
```bash
cd /Users/isaacbuz/Documents/Projects/FinTech/GoldenSignalsAI_V2
.venv/bin/python mcp_servers/simple_trading_mcp.py
```

## 📈 Sample Output

When you ask Claude to generate a signal, you'll get something like:

```json
{
  "signal_id": "AAPL_1750123456",
  "symbol": "AAPL",
  "signal_type": "BUY",
  "confidence": 0.73,
  "strength": "MODERATE",
  "current_price": 187.45,
  "reasoning": "Consensus of 19 agents with 11/19 agreeing on BUY",
  "indicators": {
    "BUY": 11,
    "SELL": 3,
    "HOLD": 5
  },
  "timestamp": "2024-12-17T16:30:45.123456",
  "summary": "Signal: BUY with 73.0% confidence (MODERATE)"
}
```

## 🎯 Next Steps (Week 2+)

1. **Connect Real Agents**: Replace simulated signals with your actual SimpleOrchestrator
2. **Add Market Data**: Create a market data MCP server
3. **Implement Security**: Add authentication and rate limiting
4. **Enhanced Features**: Add more sophisticated analysis tools

## 🎉 Congratulations!

You've successfully implemented Week 1 of the MCP integration! Your GoldenSignalsAI trading signals are now accessible through Claude Desktop using natural language commands.

Try asking Claude questions like:
- "What's your trading recommendation for Tesla?"
- "Show me signals for all tech stocks"
- "Which agents are most bullish on SPY?"

The MCP server will handle these requests and provide detailed trading insights! 