# Golden Eye AI Prophet Chat Implementation

## Overview

The Golden Eye AI Prophet Chat is a sophisticated multi-LLM system that orchestrates 9 specialized trading agents through the Model Context Protocol (MCP). It provides natural language interaction with live chart updates and intelligent trading analysis.

## Architecture

### Core Components

1. **MCP Agent Tools** (`src/services/mcp_agent_tools.py`)
   - Transforms all 9 agents into standardized MCP tools
   - Provides consensus mechanisms (weighted, majority, unanimous)
   - Enables workflow execution for complex analysis
   - Implements caching for performance

2. **Golden Eye Orchestrator** (`src/services/golden_eye_orchestrator.py`)
   - Routes queries to specialized LLMs based on intent
   - Manages streaming responses with tool execution
   - Handles chart actions and live updates
   - Implements intent classification

3. **API Routes** (`src/api/golden_eye_routes.py`)
   - RESTful endpoints for queries and discovery
   - WebSocket support for real-time chat
   - Server-Sent Events for streaming responses
   - Rate limiting and authentication

4. **Frontend Components** (`frontend/src/components/GoldenEyeChat/`)
   - React components with Material-UI
   - Real-time message streaming
   - Agent status visualization
   - Chart action integration

## LLM Specialization

### 1. **Grok 4** - Real-Time Market Intelligence
- **Specialties**: Live news, sentiment analysis, market mood
- **Agents**: SentimentAgent, MarketRegimeAgent
- **Use Cases**:
  - "What's the current market sentiment for AAPL?"
  - "Any breaking news affecting TSLA?"

### 2. **Claude 3 Opus** - Risk Assessment & Analysis
- **Specialties**: Comprehensive risk analysis, long-context reasoning
- **Agents**: OptionsChainAgent, MarketRegimeAgent, VolumeAgent
- **Use Cases**:
  - "Analyze the risk factors for entering NVDA"
  - "Explain the options flow for AMZN"

### 3. **OpenAI GPT-4o** - Technical Analysis & Predictions
- **Specialties**: Pattern recognition, technical indicators, predictions
- **Agents**: RSIAgent, MACDAgent, PatternAgent, LSTMForecastAgent
- **Use Cases**:
  - "Predict MSFT price for next 24 hours"
  - "Show technical analysis for BTC"

## MCP Tool Categories

### Technical Analysis Tools
- `analyze_with_rsiagent` - RSI overbought/oversold analysis
- `analyze_with_macdagent` - MACD trend analysis
- `analyze_with_patternagent` - Chart pattern detection
- `analyze_with_volumeagent` - Volume analysis
- `analyze_with_momentumagent` - Price momentum

### Forecasting Tools
- `analyze_with_lstmforecastagent` - Neural network predictions
- `predict_with_ai` - Ensemble AI predictions

### Market Analysis Tools
- `analyze_with_sentimentagent` - Market sentiment
- `analyze_with_marketregimeagent` - Market state detection
- `analyze_with_optionschainagent` - Options flow analysis

### Orchestration Tools
- `get_agent_consensus` - Multi-agent voting
- `execute_agent_workflow` - Complex workflow execution

## Integration Guide

### Backend Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Environment Variables**
```env
# AI Providers
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
XAI_API_KEY=your_key

# Redis
REDIS_URL=redis://localhost:6379/0

# Database
DATABASE_URL=postgresql://user:pass@localhost/goldensignals
```

3. **Start Backend**
```bash
python src/main.py
```

The Golden Eye routes are automatically included at `/api/v1/golden-eye/`

### Frontend Integration

1. **Install Golden Eye Chat**
```tsx
import { GoldenEyeChat } from './components/GoldenEyeChat';
```

2. **Basic Integration**
```tsx
const handleChartAction = (action: ChartAction) => {
  switch (action.type) {
    case 'draw_prediction':
      // Update your chart with prediction
      break;
    case 'add_agent_signals':
      // Add agent signals to chart
      break;
    // ... handle other actions
  }
};

<GoldenEyeChat
  currentSymbol="AAPL"
  onChartAction={handleChartAction}
  chartTimeframe="1h"
/>
```

3. **With UnifiedDashboard**
```tsx
// In your dashboard layout
<Grid container spacing={2}>
  <Grid item xs={8}>
    <AITradingChart ref={chartRef} />
  </Grid>
  <Grid item xs={4}>
    <GoldenEyeChat
      currentSymbol={selectedSymbol}
      onChartAction={handleChartAction}
    />
  </Grid>
</Grid>
```

## API Endpoints

### REST Endpoints

- `POST /api/v1/golden-eye/query` - Submit query (returns query ID)
- `POST /api/v1/golden-eye/query/stream` - Stream query response (SSE)
- `GET /api/v1/golden-eye/agents/discover` - Discover available agents
- `GET /api/v1/golden-eye/tools/discover` - Discover MCP tools
- `POST /api/v1/golden-eye/agents/consensus` - Get agent consensus
- `POST /api/v1/golden-eye/predict` - Generate AI prediction
- `POST /api/v1/golden-eye/workflow/{name}` - Execute workflow

### WebSocket

- `WS /api/v1/golden-eye/ws/{client_id}` - Real-time chat connection

## Usage Examples

### Natural Language Queries

1. **Prediction Request**
   - Query: "Predict AAPL price for next 24 hours"
   - LLM: OpenAI GPT-4o
   - Agents: LSTMForecastAgent, PatternAgent
   - Result: Chart with prediction line and confidence bands

2. **Technical Analysis**
   - Query: "Show me the technical setup for TSLA"
   - LLM: OpenAI GPT-4o
   - Agents: RSIAgent, MACDAgent, PatternAgent, VolumeAgent
   - Result: Consensus analysis with entry/exit recommendations

3. **Risk Assessment**
   - Query: "What are the risks of buying NVDA now?"
   - LLM: Claude 3 Opus
   - Agents: OptionsChainAgent, MarketRegimeAgent
   - Result: Detailed risk analysis with mitigation strategies

4. **Market Sentiment**
   - Query: "What's the sentiment around crypto today?"
   - LLM: Grok 4
   - Agents: SentimentAgent
   - Result: Real-time sentiment analysis with news context

### Workflow Examples

1. **Full Analysis Workflow**
```javascript
// Executes complete technical + sentiment + forecast analysis
await mcpClient.executeWorkflow('full_analysis', 'AAPL');
```

2. **Entry Signal Workflow**
```javascript
// Consensus for entry timing
await mcpClient.executeWorkflow('entry_signal', 'MSFT');
```

## Chart Actions

The Golden Eye can perform these actions on your chart:

1. **draw_prediction** - Draws AI prediction with confidence bands
2. **add_agent_signals** - Adds buy/sell signals from agents
3. **mark_entry_point** - Marks optimal entry point
4. **mark_exit_point** - Marks optimal exit point
5. **draw_levels** - Draws support/resistance levels
6. **highlight_pattern** - Highlights detected chart patterns

## Performance Optimizations

1. **Caching**
   - Agent results cached for 5 minutes
   - LLM responses cached for 1 minute
   - Redis-based distributed cache

2. **Parallel Execution**
   - Agents run in parallel for consensus
   - Multiple LLMs can be queried simultaneously

3. **Streaming**
   - Server-Sent Events for response streaming
   - WebSocket for bidirectional communication
   - Incremental UI updates

## Security

1. **Authentication**
   - JWT tokens required for all endpoints
   - User context passed to orchestrator

2. **Rate Limiting**
   - 60 requests per minute per user
   - Configurable limits per endpoint

3. **Input Validation**
   - All queries sanitized
   - Tool parameters validated
   - MCP provides sandboxed execution

## Monitoring

1. **LangSmith Integration**
   - All LLM calls traced
   - Agent execution tracked
   - Performance metrics collected

2. **Error Tracking**
   - Sentry integration for error monitoring
   - Detailed error logs with context

3. **Usage Analytics**
   - Query patterns tracked
   - Agent performance monitored
   - Cost tracking per LLM

## Testing

### Backend Tests
```bash
# Run MCP agent tests
pytest tests/test_mcp_agent_tools.py

# Test Golden Eye orchestrator
pytest tests/test_golden_eye_orchestrator.py

# Integration tests
pytest tests/test_golden_eye_integration.py
```

### Frontend Tests
```bash
# Component tests
npm test -- GoldenEyeChat

# E2E tests with Cypress
npm run cypress:open
```

## Troubleshooting

### Common Issues

1. **LLM Rate Limits**
   - Solution: Implement exponential backoff
   - Use caching to reduce API calls

2. **WebSocket Disconnections**
   - Solution: Automatic reconnection with backoff
   - Fallback to SSE if WebSocket fails

3. **Agent Timeouts**
   - Solution: Parallel execution with timeout
   - Return partial results on timeout

### Debug Mode

Enable debug logging:
```python
# In golden_eye_orchestrator.py
logger.setLevel(logging.DEBUG)
```

View agent execution:
```python
# Enable MCP tool tracing
os.environ['MCP_DEBUG'] = 'true'
```

## Future Enhancements

1. **Voice Interface**
   - Speech-to-text for queries
   - Audio responses for alerts

2. **Multi-Modal Analysis**
   - Chart screenshot analysis
   - Visual pattern recognition

3. **Advanced Workflows**
   - Custom workflow builder
   - Conditional execution paths

4. **Fine-Tuning**
   - Agent-specific model fine-tuning
   - Personalized recommendations

## Cost Estimation

### Per Query Costs (Average)
- Simple query: $0.02-0.05
- Complex analysis: $0.10-0.20
- Full workflow: $0.30-0.50

### Optimization Tips
1. Use caching aggressively
2. Batch similar queries
3. Implement query deduplication
4. Use appropriate LLM for task

## Conclusion

The Golden Eye AI Prophet Chat represents a revolutionary approach to AI-powered trading analysis. By combining specialized LLMs with domain-expert agents through the MCP protocol, it provides institutional-grade analysis through natural language conversation.

The system is designed to be:
- **Intelligent**: Multi-LLM reasoning with agent expertise
- **Interactive**: Real-time chat with live chart updates
- **Scalable**: Distributed architecture with caching
- **Extensible**: Easy to add new agents and workflows

Start with simple queries and gradually explore the full capabilities of the system.
