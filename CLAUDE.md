# Claude Instructions for GoldenSignalsAIv4

## 🚨 IMPORTANT: Development Philosophy & SOP
1. **Start Simple**: Always fix the immediate issue before adding features
2. **Don't Over-Engineer**: Ask "What's the simplest solution?" before building
3. **Incremental Changes**: Make small, testable improvements
4. **Fix, Don't Replace**: Improve existing code rather than parallel implementations
5. **User First**: Focus on solving the user's actual problem, not what you think they need

## Standard Operating Procedure (SOP) for Coding

### Before Starting Any Task
1. **Read Existing Code First**: Always check what already exists before creating new files
2. **Use TodoWrite Tool**: Plan tasks for anything requiring 3+ steps
3. **Check for Patterns**: Follow existing code conventions and patterns
4. **Verify Dependencies**: Ensure required libraries are already in package.json/requirements.txt

### During Development
1. **No Mock Data**: Only use real data from APIs, never simulate with Math.random()
2. **Fix Root Causes**: Don't work around errors - fix them at the source
3. **Single Responsibility**: Each function/component should do one thing well
4. **Type Safety**: Always define TypeScript types, never use `any` without good reason
5. **Error Handling**: Show clear error messages to users, never fail silently

### Code Quality Standards
1. **Linting**: Run `npm run lint` and `npm run typecheck` before committing
2. **Python**: Run `black src/` and `isort src/` before committing
3. **No Console Logs**: Remove all console.log statements in production code
4. **Clean Imports**: Remove unused imports immediately
5. **Meaningful Names**: Functions and variables should be self-documenting

### Testing & Validation
1. **Test Locally**: Always test changes before marking tasks complete
2. **Check All Affected Files**: One change can break multiple components
3. **Verify WebSocket Connections**: Test real-time features thoroughly
4. **Run Quality Checks**: Use pre-commit hooks (`pre-commit run --all-files`)

### Git & GitHub Workflow
1. **Commit Messages**: Use conventional commits (feat:, fix:, docs:, etc.)
2. **Small Commits**: Each commit should be one logical change
3. **Branch Names**: Use descriptive names (fix/eslint-errors, feat/ai-predictions)
4. **Pull Requests**: Reference issue numbers, provide clear descriptions
5. **Never Commit Secrets**: Use environment variables and GitHub secrets

### Common Pitfalls to Avoid
1. **Creating Duplicate Files**: Always search for existing implementations first
2. **Using Node.js Libraries in Browser**: Winston, fs, path won't work in frontend
3. **Ignoring TypeScript Errors**: They often reveal runtime issues
4. **Assuming Context**: Components may be used outside their providers
5. **Over-Architecting**: Start with the simplest solution that works

### When Stuck or Blocked
1. **Check Error Messages**: Read the full error, not just the first line
2. **Verify File Paths**: Ensure imports and file references are correct
3. **Check Network Tab**: Verify API endpoints exist and respond
4. **Review Recent Changes**: Use `git diff` to see what changed
5. **Ask for Help**: Create detailed GitHub issues with full context

## Project Overview
This is a FinTech application for analyzing trading signals and options data. The project consists of:
- Python backend (FastAPI) in `src/`
- React TypeScript frontend in `frontend/`
- WebSocket integration for real-time data

## Development Commands

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend server
python src/main.py
```

### Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint

# Run type checking
npm run typecheck
```

## Project Structure
- `src/` - Python backend code
  - `main.py` - FastAPI application entry point
- `frontend/` - React TypeScript frontend
  - `src/components/` - React components
  - `src/services/` - API and WebSocket services
  - `src/utils/` - Utility functions
- `docs/` - Documentation
- `scripts/` - Deployment and utility scripts

## Key Components
- UnifiedDashboard - Main dashboard component
- SignalCard - Displays trading signals
- OptionsChainTable - Options chain visualization
- **AITradingChart** - Main chart component (replaced all other charts)
- WebSocket services for real-time data

## Testing
Before committing changes:
1. Run frontend linting: `cd frontend && npm run lint`
2. Run frontend type checking: `cd frontend && npm run typecheck`
3. Test the application locally

## Git Workflow
- Main branch: `main`
- Create feature branches for new work
- Run tests and linting before committing
- Use descriptive commit messages

## Important Notes
- Always check existing code patterns before making changes
- Follow the existing code style and conventions
- Test WebSocket connections when modifying real-time features
- Ensure TypeScript types are properly defined

## Current AI Implementation Status

### Implemented Technologies
1. **LangGraph** - Workflow orchestration with visual debugging
2. **MCP (Model Context Protocol)** - Standardized tool interfaces
3. **Vector Database** - ChromaDB + FAISS for semantic memory
4. **LangSmith** - Observability and performance tracking
5. **Guardrails AI** - Output validation and safety
6. **Streaming LLM** - Real-time responses from multiple providers
7. **Enhanced WebSockets** - Room-based subscriptions
8. **A/B Testing** - Strategy testing with statistical significance

### Multi-LLM Setup
- GPT-4o (OpenAI) - Pattern recognition and technical analysis
- Claude 3 Opus (Anthropic) - Risk assessment and long-context analysis
- Grok 4 (xAI) - Real-time sentiment and market news

## Future Development Steps

### Advanced AI Integration
1. **CrewAI** - Multi-agent orchestration with roles and delegation
   - Implement specialized agent teams for different market conditions
   - Add inter-agent communication and task delegation

2. **AutoGen** - Microsoft's agent framework with code execution
   - Enable agents to write and execute analysis code
   - Implement self-improving strategies

3. **Pinecone/Weaviate** - Dedicated vector databases for scale
   - Migrate from ChromaDB for production scale
   - Implement distributed vector search

### Enhanced Capabilities
4. **Voice Trading Interface** - Natural language commands
   - Integrate speech-to-text for voice commands
   - Add text-to-speech for trade notifications

5. **Multi-modal Analysis** - Chart image understanding
   - Implement vision models for chart pattern recognition
   - Add screenshot analysis capabilities

6. **Real-time Strategy Adaptation** - Dynamic parameter tuning
   - Implement online learning for strategy parameters
   - Add automated backtesting and optimization

### Infrastructure Improvements
7. **Kubernetes Deployment** - Container orchestration
   - Create Helm charts for deployment
   - Implement auto-scaling based on load

8. **Distributed Processing** - Apache Kafka/Pulsar
   - Add message queue for reliable event processing
   - Implement distributed agent execution

9. **Edge Computing** - Low-latency execution
   - Deploy agents closer to exchange endpoints
   - Implement local caching and processing

## Recent Updates (Chart Implementation)

### Custom Chart Implementation
- **Replaced TradingView** with custom canvas-based charts
- **StreamlinedGoldenChart** - Main component with integrated search and dropdowns
- **EnhancedCustomChart** - Canvas rendering with animations and gradients
- **ChartSettingsMenu** - Consolidated settings in dropdown menu

### UI/UX Improvements
1. **Search Bar** - Autocomplete symbol search integrated into chart header
2. **Dropdown Timeframe** - Replaced toggle buttons (1m to 1M options)
3. **Settings Menu** - All chart options in one organized dropdown
4. **Removed Duplicates** - Cleaned up duplicate containers
5. **Professional Watermark** - Shows symbol + "GoldenSignalsAI"

### Visual Features
- Gradient backgrounds with theme support
- Glowing price lines with shadow effects
- Rounded candlesticks with gradient wicks
- Smooth animations (1.5s entry)
- Interactive buy/sell signals with pulse effect
- Real-time crosshair with price labels

### Technical Indicators Available
- RSI with divergence detection
- MACD with signal line and histogram
- Stochastic Oscillator (%K and %D)
- ATR (Average True Range)
- ADX with +DI/-DI
- Moving Averages (SMA/EMA)
- Bollinger Bands
- Volume bars with gradients

### Settings & Persistence
- Chart settings saved to localStorage
- Save/load chart layouts
- Favorite symbols management
- Theme persistence

### Current State
- Multi-timeframe tab disabled (nice to have but not necessary)
- Comparison mode still available
- All settings accessible via dropdown menu
- Clean, professional interface

## Latest Major Update: AI-Powered Chart Revolution

### Chart as the Heartbeat of the Project
The chart has been completely redesigned to be the center of excellence with highly accurate AI functionality.

### Key Improvements Implemented

#### 1. **Chart Interface Enhancements**
- **Main/Compare Tabs**: Switch between single symbol analysis and multi-symbol comparison
- **Analyze Button**: Clear call-to-action for AI analysis (replaces auto-search)
- **AI Accuracy Display**: Floating panel showing historical accuracy and confidence
- **Clean Chart**: Removed all text overlays, watermarks, and clutter
- **Indicator Dropdown**: Multi-select for choosing which indicators to display

#### 2. **Real-Time Updates Without Disruption**
- **WebSocket Integration**: Live price updates every second
- **Differential Updates**: Only price changes, no full chart redraws
- **Smooth Animations**: Price transitions without flickering
- **Connection Status**: "LIVE" indicator with pulse animation
- **Subtle Loading**: Small spinner in corner instead of full-screen

#### 3. **Advanced AI Prediction System**

##### Backend: `advanced_ai_predictor.py`
Implements ensemble prediction with multiple techniques:
- **Technical Analysis (40% weight)**: 20+ indicators including RSI, MACD, Bollinger Bands
- **Machine Learning (30% weight)**: Random Forest model trained on recent data
- **Market Microstructure (20% weight)**: Order flow, spread analysis, market depth
- **Sentiment Analysis (10% weight)**: News and social sentiment integration

##### Key Features:
- **Confidence Intervals**: Upper/lower bounds on predictions
- **Support/Resistance Detection**: Dynamic key levels calculation
- **Risk Scoring**: 0-1 scale for position sizing
- **Trend Analysis**: Multi-factor trend determination
- **Pattern Recognition**: Head & shoulders, triangles (max 3 shown)
- **Detailed Reasoning**: Explains why predictions were made

##### Accuracy Improvements:
- Historical accuracy tracking per symbol
- Ensemble predictions reduce single-model bias
- Market microstructure adds institutional perspective
- Confidence bounds show prediction uncertainty

#### 4. **Frontend Services**

##### `aiPredictionService.ts`
- Calls backend `/api/v1/ai/predict/{symbol}` endpoint
- Caches predictions for 1 minute
- Falls back to technical analysis if backend fails
- Formats predictions for chart display

##### `useRealtimeChart.ts` Hook
- Manages WebSocket connection
- Buffers price updates to prevent overwhelming
- Falls back to polling if WebSocket fails
- Smooth price update animations

#### 5. **Chart Component Updates**

##### `AITradingChart.tsx`
- Tabs for Main/Compare modes
- Search + Analyze button combo
- AI accuracy badge display
- Compare mode overlay for multiple symbols
- Smart redraw logic (only when necessary)
- Removed pattern label text from chart
- Added indicator selection dropdown

### User Experience Flow
1. User enters symbol (e.g., "AAPL")
2. Clicks "Analyze" button
3. Backend runs comprehensive AI analysis
4. Chart displays:
   - AI prediction line with confidence bounds
   - Support/resistance levels
   - High-confidence buy/sell signals only
   - AI accuracy metrics (75-85% typical)
5. Real-time price updates without disruption
6. Can switch to Compare tab to analyze multiple symbols

### Technical Implementation Details

#### Backend API Changes
- New endpoint: `POST /api/v1/ai/predict/{symbol}`
- Returns comprehensive prediction object
- Tracks accuracy history per symbol
- Uses yfinance for data, falls back to mock data

#### CORS & Rate Limiting
- Added port 5174 to CORS origins
- Increased rate limits:
  - Signal generation: 60/min (was 10)
  - Market data: 600/min (was 300)
  - Default: 500/min (was 200)

#### Performance Optimizations
- Chart only redraws when AI elements change
- Price updates modify last candle only
- Predictions cached for 1 minute
- WebSocket reduces polling overhead

### Why This Solves the Fundamental Problem

1. **Accuracy**: Not random - uses real market data and multiple AI techniques
2. **Transparency**: Shows confidence levels and historical performance
3. **Usability**: Clear analyze action, no auto-triggering confusion
4. **Professional**: Clean interface, no gimmicks or excessive patterns
5. **Real-time**: Live updates without disrupting analysis
6. **Trustworthy**: Explains reasoning, shows risk scores

### Future Enhancements Possible
- Voice commands for analysis
- Multi-modal chart image analysis
- Automated strategy backtesting
- Social sentiment integration
- Options flow analysis overlay
- Level 2 market depth visualization

## Latest Update: Beautiful Canvas Theme & Professional Trading Features

### Chart Visual Enhancements
1. **Seamless Background Integration**
   - Removed chart boundaries for seamless merge with background
   - Transparent container with subtle radial gradient overlay
   - Professional trading terminal appearance

2. **Enhanced Watermark**
   - Larger 8rem symbol size with golden gradient
   - Drop shadow glow effect (30px blur)
   - SF Pro Display font for premium feel
   - "GoldenSignalsAI" branding with letter spacing

3. **Beautiful Dark Theme**
   - Golden accent colors throughout (#FFD700)
   - Gradient backgrounds with depth
   - Glass morphism effects on headers
   - Floating gradient animations

4. **Candlestick Improvements**
   - Reduced brightness (60% opacity) for pattern visibility
   - Green candles: #00FF88 → #00CC66 gradient
   - Red candles: #FF4444 → #CC0000 gradient
   - Subtle 2px glow on wicks

5. **Grid System**
   - Golden-tinted grid lines in dark mode
   - Dashed lines with gradient opacity
   - Special emphasis on middle horizontal line
   - 12 vertical and 8 horizontal divisions

### Functional Improvements

1. **Volume Display**
   - Always visible at bottom of chart
   - 80px height with gradient bars
   - Golden gradient in dark mode
   - Semi-transparent background panel
   - "Volume" label in top-left corner

2. **Time/Date Scale**
   - Dynamically adjusts format based on timeframe:
     - 1m/5m/15m: Hour:Minute:Second
     - 1h/4h: Month Day, Hour
     - 1d: Month Day, Year
   - Golden text color in dark mode

3. **AI Prediction Line**
   - Golden gradient (#FFD700 → #FFA500 → #FF6B6B)
   - 15px glow effect with dashed line
   - Confidence bounds visualization (10% opacity fill)
   - Extends based on timeframe:
     - 1m: Next 60 minutes
     - 5m: Next hour (12 candles)
     - 1h: Next 6 hours
     - 1d: Next 5 days
   - Uses ensemble prediction with RSI, support/resistance, and volatility

4. **Technical Indicators**
   - Moving Averages (SMA/EMA) with 5px glow
   - Bollinger Bands with gradient fill
   - All indicators use elegant colors with transparency
   - Dropdown selector for easy toggle

5. **Pattern Detection**
   - Glowing effect on detected patterns
   - Golden gradient overlay (30% opacity)
   - Pulsing border animation
   - Subtle notification in top-right corner
   - Pattern type and confidence displayed

6. **Real-Time Movement**
   - Update intervals based on timeframe:
     - 1m: Every second
     - 5m: Every 5 seconds
     - 1h/4h: Every 10 seconds
     - 1d: Every 30 seconds
   - Smooth price transitions
   - Last candle updates without full redraw
   - Simulated micro-movements between data fetches

7. **Entry/Exit Signals**
   - Beautiful glowing arrows (12px size)
   - 20px shadow blur for visibility
   - Confidence rings around arrows
   - Better arrow shapes with gradient fills
   - Position offset for clarity

### UX Improvements

1. **Visual Hierarchy**
   - Clear focus on price action
   - Subtle indicators that don't overwhelm
   - Pattern notifications that inform without distraction
   - Smooth animations throughout

2. **Performance**
   - Differential rendering (only update what changes)
   - Cached calculations for indicators
   - WebSocket for real-time updates
   - Smart redraw logic

3. **Professional Feel**
   - Consistent golden accent theme
   - High-contrast text for readability
   - Smooth transitions and animations
   - Trading terminal aesthetic

4. **User Control**
   - Indicator selection dropdown
   - Chart type selector
   - Timeframe dropdown
   - Compare mode for multiple symbols
   - Analyze button for AI predictions

### Technical Architecture

1. **Canvas Rendering**
   - Dual canvas approach (main + AI overlay)
   - Hardware acceleration
   - Efficient redraw cycles
   - Smooth animations at 60fps

2. **State Management**
   - React hooks for data flow
   - Memoized calculations
   - Efficient re-renders
   - WebSocket state handling

3. **AI Integration**
   - Backend ensemble predictions
   - Frontend caching (1 minute)
   - Fallback mechanisms
   - Error boundaries

This chart implementation represents the heartbeat of GoldenSignalsAI - a professional, accurate, and beautiful trading analysis platform that provides real value to traders through advanced AI predictions and elegant visualization.

## Latest Updates: Real Data Only & Chart Organization

### 1. **Removed All Mock Data** (January 2025)
- **No more simulated data**: Chart only uses real market data from backend
- **No fake price movements**: Removed all `Math.random()` simulations
- **Error handling**: Shows clear error message when backend unavailable
- **Retry functionality**: Button to retry fetching data
- **Data integrity**: Never mixes real and fake data

#### Files Updated:
- `AITradingChart.tsx`: Removed `generateMockData()` and simulated updates
- `backendMarketDataService.ts`: Throws errors instead of returning mock data
- Real-time updates now fetch actual data at appropriate intervals

### 2. **Chart Component Consolidation**
To avoid confusion, all chart components except AITradingChart have been archived:

#### Active Chart:
- **AITradingChart** (`/src/components/AIChart/AITradingChart.tsx`) - The ONLY active chart

#### Archived Components:
Located in `/src/components/_archived_charts/`:
- `Chart/` - Original basic chart
- `CustomChart/` - Custom variations
- `ProfessionalChart/` - Professional implementation
- `RealTimeChart.tsx` - From TradingSignals
- `MainTradingChart.tsx` - From Dashboard
- `ChartWrapper.tsx` - Common wrapper
- `TransformerPredictionChart.tsx` - AI transformer chart
- `AIPredictionChart.tsx` - AI prediction component

#### Documentation:
- `CHART_REFERENCE.md` - Quick reference in components folder
- `README.md` - In archive folder explaining archived components

### 3. **Volume Display Update**
- **No background**: Volume bars render directly on chart
- **Clean integration**: Seamless appearance with main chart
- **Gradient bars**: Beautiful golden gradient in dark mode

### 4. **Edge Fade Removal**
- Removed edge fade effects per user request
- Chart has clean, sharp edges
- Clear boundaries maintained

### 5. **Real-Time Data Requirements**
To use the chart with real data:
1. Start backend: `python src/main.py`
2. Configure market data API (Yahoo Finance, Alpha Vantage, etc.)
3. Chart will fetch data from `http://localhost:8000/api/v1/market-data/`
4. Updates occur at timeframe-appropriate intervals

### Important Architecture Decisions
1. **Single Chart Component**: AITradingChart is the sole chart implementation
2. **No Mock Data**: Production-ready for real market data only
3. **Error States**: Clear user feedback when data unavailable
4. **Clean Codebase**: Archived components prevent confusion

This ensures the application is professional, maintainable, and ready for production use with real market data providers.

## Agent Architecture & Integration

### Multi-Agent System Overview
The application uses a sophisticated multi-agent architecture orchestrated by LangGraph for intelligent trading decisions:

#### Core Agents (8 Specialized Agents)
1. **RSIAgent** - Analyzes Relative Strength Index for overbought/oversold conditions
2. **MACDAgent** - Tracks Moving Average Convergence Divergence for trend changes
3. **VolumeAgent** - Monitors volume patterns and anomalies
4. **MomentumAgent** - Measures price momentum and velocity
5. **PatternAgent** - Detects candlestick patterns (head & shoulders, triangles, etc.)
6. **SentimentAgent** - Analyzes market sentiment from news and social data
7. **LSTMForecastAgent** - Uses LSTM neural networks for price predictions
8. **OptionsChainAgent** - Analyzes options flow and implied volatility skew
9. **MarketRegimeAgent** - Detects market states (trending/ranging/volatile)

#### LangGraph Workflow Orchestration
The trading workflow executes in this precise sequence:

1. **Detect Market Regime** → Identifies current market state (bullish/bearish/neutral/volatile)
2. **Collect Agent Signals** → Runs all agents in parallel for maximum efficiency
3. **Search Similar Patterns** → Uses vector memory to find historical matches
4. **Build Consensus** → Weighted voting system (agents have different performance weights)
5. **Assess Risk** → Calculates position sizing and risk parameters
6. **Make Final Decision** → Validates with Guardrails AI and executes if criteria met

#### Advanced AI Predictor
Implements ensemble prediction combining:
- **Technical Analysis (40% weight)** - 20+ indicators including RSI, MACD, Bollinger Bands
- **Machine Learning (30% weight)** - Random Forest model trained on recent data
- **Market Microstructure (20% weight)** - Order flow, spread analysis, market depth
- **Sentiment Analysis (10% weight)** - News and social sentiment integration

### Chart Integration
The AITradingChart now intelligently leverages the complete agent ecosystem:

#### **Visual Features**
- **Agent Signal Overlay**: Floating panel showing all agent signals with confidence levels
- **Trading Levels**: Automatically draws entry, stop-loss, and take-profit levels
- **Risk Visualization**: Color-coded risk levels (LOW/MEDIUM/HIGH)
- **Consensus Display**: Shows weighted voting results from all agents

#### **Workflow Integration**
- **Analyze Button**: Triggers full LangGraph workflow analysis
- **Real-time Updates**: WebSocket subscriptions for live agent signals
- **Decision Validation**: All trading decisions pass Guardrails AI safety checks
- **Historical Context**: Vector memory provides similar pattern analysis

#### **API Endpoints**
- `/api/v1/workflow/analyze-langgraph/{symbol}` - Complete workflow analysis
- `/api/v1/agents/` - Agent status and performance metrics
- `/api/v1/ai/predict/{symbol}` - Advanced AI predictions
- WebSocket channels for real-time agent signals

### Agent Performance & Weights
Agents have different performance weights in consensus building:
- **LSTMForecastAgent**: 1.4 (highest weight - best historical performance)
- **PatternAgent**: 1.3 (strong pattern recognition)
- **OptionsChainAgent**: 1.25 (institutional flow insights)
- **RSIAgent**: 1.2 (reliable momentum indicator)
- **MACDAgent**: 1.1 (trend confirmation)
- **MomentumAgent**: 1.15 (velocity analysis)
- **VolumeAgent**: 1.0 (standard weight)
- **SentimentAgent**: 0.9 (lower weight due to noise)

### Key Benefits
1. **Accuracy**: Ensemble approach reduces single-model bias
2. **Transparency**: Shows individual agent reasoning and confidence
3. **Risk Management**: Built-in position sizing and stop-loss calculations
4. **Scalability**: Parallel agent execution for fast analysis
5. **Safety**: Guardrails AI prevents dangerous trading decisions
6. **Learning**: Vector memory improves decisions based on historical patterns

This agent architecture represents the core intelligence of GoldenSignalsAI, providing institutional-grade analysis through specialized AI agents working in harmony.

## Common Issues & Simple Fixes

### 1. **Huge Candlesticks**
**Problem**: Candlesticks appear unrealistically large
**Root Cause**: Data normalization issue, not a charting library problem
**Simple Fix**: Use `candleNormalizer` utility in `utils/candleNormalizer.ts`
```typescript
const normalizedData = candleNormalizer.normalizeData(data, timeframe);
```

### 2. **WebSocket Connection Issues**
**Problem**: Real-time updates not working
**Simple Fix**: Check backend is running, verify CORS settings include your port
**Don't**: Create a new WebSocket implementation
**Do**: Fix the existing one with reconnection logic

### 3. **Performance Issues**
**Problem**: Chart laggy with many indicators
**Simple Fix**: Debounce updates, use requestAnimationFrame
**Don't**: Switch to a different charting library
**Do**: Optimize the existing render cycle

### 4. **API Limits**
**Problem**: Data provider API limits exceeded
**Simple Fix**: Implement caching, use fallback providers
**Don't**: Mock the data
**Do**: Show clear error messages to user

## Best Practices

### When Adding Features
1. **Feature Flags**: Use feature flags for gradual rollout
2. **Backwards Compatible**: Never break existing functionality
3. **Document Changes**: Update this file with new patterns
4. **Test First**: Write tests before implementing

### Code Organization
1. **Single Responsibility**: Each component does one thing well
2. **Reuse Existing**: Check for existing utilities before creating new ones
3. **Clear Naming**: Functions should explain what they do
4. **Type Safety**: Always define TypeScript types

### Performance
1. **Measure First**: Profile before optimizing
2. **Cache Wisely**: Use localStorage for user preferences, not data
3. **Batch Updates**: Group multiple state changes
4. **Lazy Load**: Load heavy components only when needed

## Critical Bug Fixes & Session Learnings (January 2025)

### Major Issues Fixed in Recent Session

#### 1. **Import Statement Syntax Errors**
**Problem**: Broken import statements in multiple files causing 500 errors
**Files Affected**:
- `useAgentWebSocket.ts` - Logger import was split incorrectly
- `agentWorkflowService.ts` - Logger import was split incorrectly
**Fix**: Properly separated imports:
```typescript
import logger from './logger';
import { TypeA, TypeB } from './types';
```

#### 2. **WebSocket Endpoint Mismatches**
**Problem**: Frontend trying to connect to non-existent WebSocket endpoints
**Root Cause**: Frontend was using `/ws/agents/{symbol}` but backend only has `/ws/v2/signals/{symbol}`
**Files Fixed**:
- `useAgentWebSocket.ts` - Changed to use `/ws/v2/signals/${symbol}`
- `agentWorkflowService.ts` - Changed to use `/ws/v2/signals/${symbol}`
**Backend WebSocket Endpoints Available**:
- `/ws` - Basic WebSocket connection
- `/ws/market-data/{symbol}` - Market data streaming
- `/ws/v2/connect` - V2 connection
- `/ws/v2/signals/{symbol}` - V2 signals streaming

#### 3. **Duplicate Imports in ChartCanvas**
**Problem**: LayerManager and other imports were duplicated causing "already declared" errors
**File**: `ChartCanvas.tsx`
**Fix**: Removed duplicate import lines (lines 32-34 were duplicates of 29-31)

#### 4. **Winston Logger in Browser**
**Problem**: Winston (Node.js library) cannot run in browser environment
**Error**: `util.inherits is not a function`
**Solution**: Replaced Winston with custom browser-compatible logger
**File**: `frontend/src/services/logger.ts`
**Features of New Logger**:
- Browser-compatible (no Node.js dependencies)
- Colored console output in development
- Intercepts console.log calls
- Maintains same API as Winston
- Lightweight with no external dependencies

#### 5. **Variable Reference Errors**
**Problem**: Multiple undefined variable references
**Fixes**:
- `isConnected` → `isAgentConnected` in `AITradingChart.tsx` (lines 313, 322)
- Added missing `timeframe` prop to `ChartCanvas` component
- Added missing mouse event handlers to `ChartCanvas`

#### 6. **Context Usage Issues**
**Problem**: ChartCanvas trying to use ChartContext but not always within provider
**Solution**: Added `timeframe` as a prop instead of getting from context
**Pattern**: When components might be used outside their context provider, accept critical values as props with defaults

### Application Architecture Understanding

#### Frontend Structure
```
frontend/src/
├── components/
│   ├── AIChart/           # Main chart component (ONLY active chart)
│   │   ├── AITradingChart.tsx
│   │   ├── components/
│   │   │   └── ChartCanvas/
│   │   ├── context/       # Chart context provider
│   │   ├── hooks/         # Chart-specific hooks
│   │   └── utils/         # Chart utilities
│   └── _archived_charts/  # All other chart implementations (archived)
├── hooks/
│   ├── useAgentWebSocket.ts    # WebSocket for agent signals
│   ├── useAgentAnalysis.ts     # Agent workflow analysis
│   └── useRealtimeChart.ts     # Real-time price updates
├── services/
│   ├── logger.ts               # Custom browser logger
│   ├── agentWorkflowService.ts # Agent API integration
│   └── websocket/              # WebSocket services
└── pages/
    └── EnhancedTradingDashboard.tsx # Main dashboard
```

#### Backend Structure
```
src/
├── main.py                # FastAPI app with WebSocket endpoints
├── services/
│   ├── advanced_ai_predictor.py  # Ensemble AI predictions
│   ├── golden_eye_orchestrator.py # LangGraph workflow
│   └── websocket_manager.py      # WebSocket management
└── middleware/
    └── rate_limiter.py    # Rate limiting (fail-closed)
```

#### Data Flow
1. **User Action**: Enter symbol → Click "Analyze"
2. **Frontend**: Calls `/api/v1/workflow/analyze-langgraph/{symbol}`
3. **Backend**: Runs LangGraph workflow with 9 agents
4. **WebSocket**: Real-time updates via `/ws/v2/signals/{symbol}`
5. **Chart**: Updates with predictions, levels, and signals

#### Key Technologies
- **Frontend**: React, TypeScript, Material-UI, Canvas rendering
- **Backend**: FastAPI, LangGraph, Multiple LLMs (GPT-4, Claude, Grok)
- **Real-time**: WebSockets with reconnection logic
- **State**: React Context + hooks for chart state

### Development Guidelines

#### When Debugging Frontend Errors
1. **Check Browser Console First**: Look for syntax errors, undefined variables
2. **Verify Imports**: Ensure imports are properly formatted
3. **Check Network Tab**: Verify API endpoints exist and respond
4. **Validate Props**: Ensure all required props are passed
5. **Test in Isolation**: Try components with minimal setup

#### Common Pitfalls to Avoid
1. **Don't use Node.js libraries in browser** (like Winston, fs, path)
2. **Don't assume context is always available** - provide fallbacks
3. **Don't create new implementations** - fix existing ones
4. **Don't ignore TypeScript errors** - they often reveal runtime issues

#### WebSocket Best Practices
1. **Always check endpoint exists** in backend before using
2. **Implement reconnection logic** with exponential backoff
3. **Handle connection states** (connecting, connected, error)
4. **Clean up on unmount** to prevent memory leaks

### Environment Setup

#### Required Services
1. **Backend**: `python src/main.py` (port 8000)
2. **Frontend**: `npm run dev` (port 3000)
3. **Redis**: For rate limiting and caching
4. **PostgreSQL**: For data persistence (optional)

#### API Keys Required
- OpenAI API Key (GPT-4)
- Anthropic API Key (Claude)
- xAI API Key (Grok)
- Market data provider (TwelveData, Finnhub, etc.)

#### Browser Access
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Remember
The goal is to build a reliable, professional trading platform. Every line of code should serve the user's needs, not showcase technical complexity. When errors occur, fix the root cause rather than working around it.
