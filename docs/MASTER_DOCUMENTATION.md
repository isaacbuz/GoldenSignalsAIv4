# GoldenSignalsAI Master Documentation

**Last Updated:** December 2024  
**Version:** 4.0

## Table of Contents

### [PART I: PROJECT OVERVIEW](#part-i-project-overview)
- [1. Introduction](#1-introduction)
- [2. Core Features](#2-core-features)
- [3. System Architecture](#3-system-architecture)
- [4. Technology Stack](#4-technology-stack)
- [5. Project Structure](#5-project-structure)

### [PART II: GETTING STARTED](#part-ii-getting-started)
- [6. Prerequisites](#6-prerequisites)
- [7. Installation](#7-installation)
- [8. Configuration](#8-configuration)
- [9. Running the Application](#9-running-the-application)

### [PART III: DEVELOPMENT GUIDE](#part-iii-development-guide)
- [10. Development Setup](#10-development-setup)
- [11. Component Library](#11-component-library)
- [12. Domain Models](#12-domain-models)
- [13. Testing](#13-testing)

### [PART IV: API REFERENCE](#part-iv-api-reference)
- [14. API Overview](#14-api-overview)
- [15. Signal Generation](#15-signal-generation)
- [16. Market Data](#16-market-data)
- [17. Signal Monitoring](#17-signal-monitoring)
- [18. WebSocket API](#18-websocket-api)

### [PART V: OPERATIONS & DEPLOYMENT](#part-v-operations--deployment)
- [19. Performance Monitoring](#19-performance-monitoring)
- [20. Troubleshooting](#20-troubleshooting)
- [21. Future Enhancements](#21-future-enhancements)

---

## PART I: PROJECT OVERVIEW

### 1. Introduction

GoldenSignalsAI is a production-ready, enterprise-grade AI-powered trading signal intelligence platform. The system leverages advanced machine learning, multi-agent consensus algorithms, and real-time data processing to generate high-quality trading signals for both retail and institutional traders.

#### Mission
To democratize access to sophisticated trading intelligence by providing AI-driven insights that were previously available only to large financial institutions.

#### Vision
To become the leading AI-powered trading assistant platform, helping traders make informed decisions with confidence.

#### Key Statistics
- **Signal Generation**: <100ms average response time
- **WebSocket Latency**: <10ms for real-time updates
- **API Response Time**: <50ms (p95)
- **Test Coverage**: >80%
- **Uptime**: 99.9% SLA

### 2. Core Features

#### 2.1 Multi-Agent Trading System
- **30+ Specialized Agents**: Each focusing on different aspects of market analysis
- **Byzantine Fault Tolerance**: Consensus mechanism ensures signal reliability
- **CrewAI Integration**: Advanced agent orchestration and coordination
- **Real-time Performance Tracking**: Monitor each agent's accuracy and contribution

#### 2.2 Signal Generation
- **Multi-Timeframe Analysis**: From 1-minute to daily signals
- **Asset Coverage**: Stocks, Options, Crypto, Forex, Commodities
- **Confidence Scoring**: Each signal includes AI confidence metrics
- **Risk Assessment**: Integrated risk analysis for every signal

#### 2.3 Real-Time Data Processing
- **WebSocket Streaming**: Sub-10ms latency for market data
- **Multiple Data Sources**: Yahoo Finance, Polygon.io, Alpha Vantage
- **Smart Caching**: Redis-based caching for optimal performance
- **Rate Limit Management**: Intelligent handling of API limits

#### 2.4 AI/ML Capabilities
- **Transformer Models**: State-of-the-art deep learning for pattern recognition
- **RAG (Retrieval-Augmented Generation)**: Context-aware signal generation
- **Sentiment Analysis**: Real-time news and social media analysis
- **Adaptive Learning**: Models improve from market feedback

#### 2.5 Professional UI/UX
- **React 18 + TypeScript**: Modern, type-safe frontend
- **TradingView-style Charts**: Professional-grade visualizations
- **Dark Theme**: Optimized for extended trading sessions
- **Responsive Design**: Works on desktop, tablet, and mobile

#### 2.6 Enterprise Features
- **Horizontal Scaling**: Supports 10+ nodes with Redis pub/sub
- **Monitoring & Observability**: Prometheus, Grafana, OpenTelemetry
- **Security**: JWT auth, rate limiting, CORS protection
- **CI/CD Pipeline**: Automated testing and deployment

### 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React/TypeScript)               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ Dashboard  │  │  AI Lab    │  │  Signals   │  │ Analytics│ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                           ┌────────┴────────┐
                           │   WebSocket     │
                           │   REST API      │
                           └────────┬────────┘
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI/Python)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   Agent    │  │   Signal   │  │   Market   │  │    AI    │ │
│  │Orchestrator│  │  Service   │  │Data Service│  │ Service  │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
           ┌────────┴────────┐  ┌──┴───┐  ┌────────┴────────┐
           │   PostgreSQL    │  │Redis │  │   ChromaDB      │
           │   (Primary DB)  │  │Cache │  │ (Vector Store)  │
           └─────────────────┘  └──────┘  └─────────────────┘
```

#### 3.1 Architectural Principles
- **Modular Design**: Clear separation of concerns
- **Microservices Architecture**: Independent scaling and deployment
- **Event-Driven**: Pub/sub for real-time updates
- **Type Safety**: TypeScript and Python type hints throughout
- **Comprehensive Testing**: Unit, integration, and E2E tests

#### 3.2 Directory Structure

##### Root Level
- `agents/`: Multi-agent trading signal generation system
- `infrastructure/`: Core system infrastructure
- `services/`: Business logic implementations
- `models/`: Data models and schemas
- `tests/`: Comprehensive test suite
- `frontend/`: React-based user interface
- `docs/`: Documentation

##### Core Components
- **Application Layer**: Core business logic and services
- **Domain Layer**: Data models and trading logic
- **Infrastructure Layer**: External data integration
- **Presentation Layer**: User interface and API

### 4. Technology Stack

#### 4.1 Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 15+ with asyncpg
- **Cache**: Redis 7+ for real-time data
- **Vector DB**: ChromaDB for RAG
- **Task Queue**: Celery with Redis broker
- **ML Framework**: PyTorch, scikit-learn, transformers

#### 4.2 Frontend
- **Framework**: React 18 with TypeScript
- **State Management**: Zustand + TanStack Query
- **UI Library**: Material-UI (MUI) v5
- **Charts**: Lightweight Charts (TradingView)
- **Build Tool**: Vite
- **Testing**: Jest, React Testing Library, Cypress

#### 4.3 Infrastructure
- **Container**: Docker + Docker Compose
- **Orchestration**: Kubernetes (K8s)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: Structured logging with JSON
- **Cloud**: AWS/GCP/Azure compatible

### 5. Project Structure

The project follows a modular architecture with clear separation of concerns:

```
GoldenSignalsAIv4/
├── agents/                 # Multi-agent system
│   ├── predictive/        # Signal generation agents
│   ├── sentiment/         # Sentiment analysis agents
│   └── risk/             # Risk assessment agents
├── infrastructure/        # Core system infrastructure
├── services/             # Business logic implementations
├── models/              # Data models and schemas
├── frontend/            # React-based UI
├── docs/               # Documentation
└── tests/              # Comprehensive test suite
```

---

## PART II: GETTING STARTED

### 6. Prerequisites

#### 6.1 System Requirements
- **Operating System**: macOS, Linux, or Windows (with WSL2)
- **Python**: 3.8+ with pip
- **Node.js**: 16+ with npm
- **Git**: For version control
- **PostgreSQL**: 12+ (optional, for persistence)
- **Redis**: 6+ (optional, for caching)

#### 6.2 Hardware Requirements
- **RAM**: Minimum 8GB, recommended 16GB
- **CPU**: Multi-core processor recommended for ML operations
- **Storage**: At least 10GB free space for dependencies and data

### 7. Installation

#### 7.1 Quick Start (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd GoldenSignalsAIv4

# Start all services
./start.sh
```

This single command will:
- Check all prerequisites
- Create virtual environment if needed
- Install all dependencies
- Start databases (if available)
- Launch backend API
- Launch frontend UI

#### 7.2 Manual Installation

##### Backend Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

##### Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Install Storybook (for component development)
npm install --save-dev @storybook/react-vite@^8.6.0
```

##### Database Setup (Optional)
```bash
# PostgreSQL
createdb goldensignals_dev

# Run migrations
alembic upgrade head

# Redis (if using Docker)
docker run -d -p 6379:6379 redis:alpine
```

### 8. Configuration

#### 8.1 Backend Configuration
Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost/goldensignals_dev

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# External APIs (if needed)
ALPHA_VANTAGE_API_KEY=your-key
POLYGON_API_KEY=your-key
```

#### 8.2 Frontend Configuration
Create a `.env` file in the `frontend` directory:

```env
# API URL
VITE_API_URL=http://localhost:8000

# WebSocket Configuration
VITE_WEBSOCKET_ENABLED=true
VITE_WEBSOCKET_URL=ws://localhost:8000/ws

# Feature Flags
VITE_FEATURE_AI_CHAT=true
VITE_FEATURE_ADVANCED_CHARTS=false
VITE_FEATURE_PAPER_TRADING=false

# Logging
VITE_LOG_LEVEL=info
VITE_LOG_TO_FILE=true
VITE_ENABLE_DEBUG_PANEL=true

# Performance
VITE_ENABLE_PERFORMANCE_MONITORING=true
VITE_SLOW_RENDER_THRESHOLD=100
```

### 9. Running the Application

#### 9.1 Using the Master Script
```bash
# Start all services
./start.sh

# Start with options
./start.sh start --mode dev        # Development mode (default)
./start.sh start --mode docker     # Docker mode
./start.sh start --detached        # Run in background

# Service management
./start.sh stop                    # Stop all services
./start.sh restart                 # Restart all services
./start.sh status                  # Check service status
```

#### 9.2 Manual Startup

##### Backend
```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

##### Frontend
```bash
cd frontend

# Development server
npm run dev

# Production build
npm run build
npm run preview
```

#### 9.3 Accessing the Application
- **Frontend**: http://localhost:5173 (Vite dev server)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Storybook**: http://localhost:6006 (run `npm run storybook`)

---

## PART III: DEVELOPMENT GUIDE

### 10. Development Setup

#### 10.1 IDE Configuration

##### VS Code
Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)
- TypeScript and JavaScript (ms-vscode.typescript-javascript)

Create `.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "typescript.preferences.importModuleSpecifier": "relative",
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

#### 10.2 Component Development with Storybook
```bash
cd frontend

# Start Storybook
npm run storybook

# Build Storybook
npm run build-storybook
```

#### 10.3 Generating Components
Use the component generator for consistent component creation:

```bash
cd frontend
npm run generate:component <ComponentName>
```

This creates:
- Component file with TypeScript and logging
- CSS module
- Test file with standard setup
- Storybook story
- Proper exports

### 11. Component Library

#### 11.1 Overview
The GoldenSignalsAI component library provides production-ready React components built with TypeScript, featuring full observability, accessibility, and modern design patterns.

#### 11.2 Core Principles
- **Observability**: Every component logs its lifecycle and interactions
- **Accessibility**: ARIA compliance and keyboard navigation
- **Type Safety**: Full TypeScript support with detailed prop types
- **Testing**: Comprehensive test coverage with data-testid conventions
- **Responsive**: Mobile-first design approach
- **Dark Mode**: Automatic dark mode support

#### 11.3 Available Components

##### Button Component
A versatile button component with multiple variants and states.

**Features:**
- 5 variants: primary, secondary, danger, ghost, link
- 3 sizes: small, medium, large
- Icon support (left/right positioning)
- Loading states with spinner
- Ripple effect animation

**Usage:**
```tsx
import { Button } from '@/components/Core/Button';

<Button 
  variant="primary" 
  size="medium"
  loading={isLoading}
  onClick={handleClick}
  leftIcon={<SaveIcon />}
>
  Save Changes
</Button>
```

##### Input Component
A flexible input component with multiple styles and features.

**Features:**
- 3 variants: outlined, filled, standard
- Error states with messages
- Helper text support
- Character counting
- Icon support (start/end)
- Password visibility toggle

**Usage:**
```tsx
import { Input } from '@/components/Input';

<Input
  label="Email"
  type="email"
  variant="outlined"
  value={email}
  onChange={(e) => setEmail(e.target.value)}
  error={emailError}
  helperText="We'll never share your email"
  startIcon={<EmailIcon />}
  required
/>
```

##### Table Component
A feature-rich data table component.

**Features:**
- Sorting (single/multi-column)
- Filtering
- Pagination
- Row selection
- Custom cell rendering
- Sticky header

**Usage:**
```tsx
import { Table } from '@/components/Table';

<Table
  data={signals}
  columns={[
    { key: 'symbol', header: 'Symbol', sortable: true },
    { key: 'signal', header: 'Signal', filterable: true },
    { key: 'confidence', header: 'Confidence', render: (val) => `${val}%` }
  ]}
  pageSize={10}
  selectable
  onSelectionChange={setSelectedRows}
/>
```

##### Loading Component
Comprehensive loading states and skeleton screens.

**Features:**
- 6 types: spinner, dots, bars, skeleton, progress, pulse
- 3 sizes and 6 color variants
- Overlay mode
- Skeleton sub-components
- Progress tracking

**Usage:**
```tsx
import { Loading, LoadingContainer, SkeletonCard } from '@/components/Loading';

<Loading type="spinner" size="medium" text="Loading..." />
<LoadingContainer loading={isLoading}>
  <YourContent />
</LoadingContainer>
```

##### Modal Component
Flexible modal dialogs with animations.

**Features:**
- 4 sizes: small, medium, large, fullscreen
- 4 animations: fade, slide, scale, none
- Focus management
- Keyboard navigation
- Specialized variants (Confirm, Alert)

**Usage:**
```tsx
import { Modal, ConfirmModal } from '@/components/Modal';

<Modal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Edit Profile"
  size="medium"
>
  <ProfileForm />
</Modal>
```

### 12. Domain Models

#### 12.1 Overview
The domain layer contains the core business logic for the trading system, organized into four main components:
- Data Management
- Model Management
- Portfolio Management
- Analytics

#### 12.2 Data Management
The `DataManager` class handles all market data operations.

**Key Features:**
- Market data fetching with caching
- Feature preparation for machine learning
- Data splitting for model training

**Example Usage:**
```python
data_manager = DataManager()

# Fetch market data
data = data_manager.fetch_market_data(
    symbols=["AAPL", "MSFT"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Prepare features
features, target = data_manager.prepare_features(
    data=data["AAPL"],
    feature_columns=['Open', 'High', 'Low', 'Volume'],
    target_column='Close',
    lookback_periods=5
)
```

#### 12.3 Model Management
The `ModelManager` class handles machine learning model operations.

**Key Features:**
- Model training and persistence
- Prediction generation
- Model evaluation
- Automatic model saving and loading

**Example Usage:**
```python
model_manager = ModelManager(model_dir="ml_models")

# Train model
model = model_manager.train_model(
    model_id="aapl_predictor",
    X_train=features,
    y_train=target,
    model_type="regressor"
)

# Make predictions
predictions = model_manager.predict("aapl_predictor", new_features)
```

#### 12.4 Portfolio Management
The `PortfolioManager` class handles trading and portfolio operations.

**Key Features:**
- Order placement and position tracking
- Portfolio value calculation
- Position size management
- Trade history tracking

**Example Usage:**
```python
portfolio_manager = PortfolioManager(initial_capital=100000.0)

# Place order
success = portfolio_manager.place_order(
    symbol="AAPL",
    quantity=100,
    price=150.0
)

# Get portfolio metrics
metrics = portfolio_manager.get_portfolio_metrics(
    current_prices={"AAPL": 155.0}
)
```

#### 12.5 Analytics
The `AnalyticsManager` class provides financial analytics and risk metrics.

**Key Features:**
- Returns calculation (arithmetic and logarithmic)
- Risk metrics (volatility, Sharpe ratio, etc.)
- Portfolio performance metrics (alpha, beta, etc.)
- Drawdown analysis

**Example Usage:**
```python
analytics_manager = AnalyticsManager()

# Calculate returns
returns = analytics_manager.calculate_returns(prices, method='arithmetic')

# Get risk metrics
risk_metrics = analytics_manager.calculate_risk_metrics(returns)

# Analyze drawdowns
drawdowns = analytics_manager.calculate_drawdowns(returns, top_n=5)
```

### 13. Testing

#### 13.1 Backend Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run tests in watch mode
ptw
```

#### 13.2 Frontend Tests
```bash
cd frontend

# Run unit tests
npm run test

# Run with coverage
npm run test:coverage

# Run E2E tests with Cypress
npm run cypress:open
```

#### 13.3 Integration Tests
Comprehensive integration tests cover:
- End-to-end workflow
- Multi-asset portfolio management
- Model persistence
- Risk analytics

---

## PART IV: API REFERENCE

### 14. API Overview

**Base URL**: `http://localhost:8000/api/v1`  
**API Version**: v1  
**Authentication**: Currently no authentication required (development mode)

#### 14.1 Health Check
```http
GET /
```

**Response:**
```json
{
  "message": "GoldenSignalsAI API is running",
  "status": "operational",
  "timestamp": "2024-12-23T10:30:00Z",
  "uptime": 3600,
  "version": "2.0.0"
}
```

#### 14.2 Performance Metrics
```http
GET /api/v1/performance
```

**Response:**
```json
{
  "requests_per_endpoint": {
    "/api/v1/signals": 1523,
    "/api/v1/market-data/SPY": 892
  },
  "cache_stats": {
    "hits": 3421,
    "misses": 876,
    "hit_rate": "79.6%"
  },
  "response_times": {
    "average": 45.2,
    "p95": 120.5,
    "p99": 250.3
  }
}
```

### 15. Signal Generation

#### 15.1 Get All Signals
```http
GET /api/v1/signals
```

**Query Parameters:**
- `symbols` (optional): Comma-separated list of symbols
- `min_confidence` (optional): Minimum confidence threshold (0-1)
- `risk_levels` (optional): Comma-separated risk levels (low,medium,high)

**Response:**
```json
{
  "signals": [
    {
      "id": "sig_123456",
      "symbol": "AAPL",
      "action": "BUY",
      "confidence": 0.85,
      "price": 185.50,
      "indicators": {
        "RSI": 45.2,
        "MACD": 0.35,
        "BB_position": 0.3
      },
      "risk_level": "medium",
      "entry_price": 185.50,
      "stop_loss": 182.00,
      "take_profit": 190.00,
      "timestamp": "2024-12-23T10:30:00Z"
    }
  ],
  "count": 15,
  "generated_at": "2024-12-23T10:30:00Z"
}
```

#### 15.2 Get Signal Insights
```http
GET /api/v1/signals/{symbol}/insights
```

**Response:**
```json
{
  "symbol": "AAPL",
  "current_signal": {
    "action": "BUY",
    "confidence": 0.85
  },
  "technical_analysis": {
    "trend": "bullish",
    "support_levels": [180.0, 175.0],
    "resistance_levels": [190.0, 195.0],
    "indicators": {
      "RSI": {"value": 45.2, "signal": "neutral"},
      "MACD": {"value": 0.35, "signal": "bullish"}
    }
  },
  "risk_metrics": {
    "volatility": 0.023,
    "beta": 1.15,
    "max_drawdown": 0.08
  }
}
```

#### 15.3 Generate Batch Signals
```http
POST /api/v1/signals/batch
```

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
  "parameters": {
    "min_confidence": 0.7,
    "include_indicators": true
  }
}
```

### 16. Market Data

#### 16.1 Get Real-time Quote
```http
GET /api/v1/market-data/{symbol}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 185.50,
  "change": 2.30,
  "change_percent": 1.25,
  "volume": 45238910,
  "high": 186.80,
  "low": 183.20,
  "open": 184.00,
  "previous_close": 183.20,
  "timestamp": "2024-12-23T10:30:00Z"
}
```

#### 16.2 Get Historical Data
```http
GET /api/v1/market-data/{symbol}/historical
```

**Query Parameters:**
- `period`: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
- `interval`: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

**Response:**
```json
{
  "symbol": "AAPL",
  "period": "1mo",
  "interval": "1d",
  "data": [
    {
      "date": "2024-12-01",
      "open": 180.50,
      "high": 182.30,
      "low": 179.80,
      "close": 181.50,
      "volume": 50234100
    }
  ]
}
```

### 17. Signal Monitoring

#### 17.1 Track Signal Entry
```http
POST /api/v1/monitoring/track-entry
```

**Request Body:**
```json
{
  "signal_id": "sig_123456",
  "symbol": "AAPL",
  "entry_price": 185.50,
  "quantity": 100,
  "entry_time": "2024-12-23T10:30:00Z"
}
```

#### 17.2 Get Performance Metrics
```http
GET /api/v1/monitoring/performance
```

**Response:**
```json
{
  "overall": {
    "total_signals": 150,
    "win_rate": 0.68,
    "average_profit": 2.35,
    "average_loss": -1.80,
    "profit_factor": 1.85,
    "sharpe_ratio": 1.45,
    "max_drawdown": -8.5
  },
  "by_symbol": {
    "AAPL": {
      "signals": 25,
      "win_rate": 0.72,
      "avg_return": 2.8
    }
  }
}
```

### 18. WebSocket API

#### 18.1 Real-time Signal Updates
```websocket
ws://localhost:8000/ws
```

**Connection Message:**
```json
{
  "type": "subscribe",
  "channels": ["signals", "market_data"],
  "symbols": ["AAPL", "GOOGL", "SPY"]
}
```

**Signal Update:**
```json
{
  "type": "signal_update",
  "data": {
    "symbol": "AAPL",
    "action": "BUY",
    "confidence": 0.85,
    "timestamp": "2024-12-23T10:30:00Z"
  }
}
```

#### 18.2 Error Handling
All endpoints follow a consistent error response format:

```json
{
  "error": "Error message",
  "detail": "Detailed error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-12-23T10:30:00Z"
}
```

**Common Error Codes:**
- `INVALID_SYMBOL` (400): Invalid or unsupported symbol
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `DATA_UNAVAILABLE` (503): Market data temporarily unavailable
- `INVALID_PARAMETERS` (422): Invalid request parameters

---

## PART V: OPERATIONS & DEPLOYMENT

### 19. Performance Monitoring

#### 19.1 System Performance Metrics
The system provides comprehensive performance monitoring:

- **Signal Generation**: <100ms average
- **WebSocket Latency**: <10ms
- **API Response Time**: <50ms (p95)
- **Test Coverage**: >80%
- **Uptime**: 99.9%

#### 19.2 Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **OpenTelemetry**: Distributed tracing
- **Structured Logging**: JSON-formatted logs

### 20. Troubleshooting

#### 20.1 Common Issues

##### Port Already in Use
```bash
# Find and kill process on port
# macOS/Linux:
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9

# Or use the utility script:
./dev-utils.sh clean
```

##### Dependencies Not Installing
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# For Python, recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

##### WebSocket Connection Issues
- Check that backend is running on correct port
- Verify VITE_WEBSOCKET_URL in frontend .env
- Check browser console for CORS errors
- Disable WebSocket temporarily: `VITE_WEBSOCKET_ENABLED=false`

#### 20.2 Debug Mode

##### Enable Debug Logging
```bash
# Backend
export LOG_LEVEL=DEBUG
uvicorn src.main:app --reload --log-level debug

# Frontend
VITE_LOG_LEVEL=debug npm run dev
```

##### Debug Panel
The Debug Panel is available in development mode at the bottom-right corner, showing:
- Frontend errors and warnings
- API call logs
- Performance metrics
- WebSocket status

#### 20.3 Getting Help
- Check logs: `./start.sh logs [backend|frontend]`
- Run diagnostics: `./status-check.sh`
- View API docs: http://localhost:8000/docs
- Check component stories: http://localhost:6006

### 21. Future Enhancements

#### 21.1 Position Sizing and Risk Management
- Dynamic position sizing based on confidence levels and market volatility
- Kelly Criterion implementation for optimal bet sizing
- Portfolio-level risk management with correlation analysis
- Maximum position size limits based on account equity
- Risk parity allocation across multiple signals

#### 21.2 Advanced Entry/Exit Strategies
- Automated stop-loss calculation based on volatility and confidence
- Dynamic take-profit levels using support/resistance detection
- Trailing stop implementation with confidence-based adjustments
- Partial position scaling based on signal strength
- Time-based exit rules for momentum strategies

#### 21.3 Market Regime Detection
- Market regime classification (trending, ranging, volatile)
- Regime-specific signal adjustments
- Volatility regime detection and adaptation
- Correlation regime analysis with major indices
- Market sentiment integration

#### 21.4 Machine Learning Enhancements
- Ensemble methods with multiple timeframes
- Feature importance analysis
- Automated feature engineering
- Hyperparameter optimization
- Model drift detection

#### 21.5 Real-time Monitoring
- Live performance dashboard
- Alert system for significant drawdowns
- Automated strategy health checks
- Real-time risk exposure monitoring
- Performance deviation alerts

#### 21.6 Implementation Priority
1. **Position Sizing and Risk Management** (High)
2. **Advanced Entry/Exit Strategies** (High)
3. **Real-time Monitoring** (High)
4. **Market Regime Detection** (Medium)
5. **Machine Learning Enhancements** (Medium)

#### 21.7 Success Metrics
- Risk-adjusted returns (Sharpe, Sortino ratios)
- Maximum drawdown reduction
- Win rate improvement
- Risk-reward ratio enhancement
- Transaction cost reduction
- Strategy robustness (Monte Carlo simulation)

---

## APPENDICES

### A. Code Examples

#### A.1 Python API Usage
```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Get signals
response = requests.get(f"{BASE_URL}/signals")
signals = response.json()

for signal in signals["signals"]:
    print(f"{signal['symbol']}: {signal['action']} (confidence: {signal['confidence']})")

# Track signal entry
entry_data = {
    "signal_id": signals["signals"][0]["id"],
    "symbol": signals["signals"][0]["symbol"],
    "entry_price": signals["signals"][0]["price"],
    "quantity": 100,
    "entry_time": "2024-12-23T10:30:00Z"
}

response = requests.post(f"{BASE_URL}/monitoring/track-entry", json=entry_data)
print("Entry tracked:", response.json())
```

#### A.2 JavaScript/React Usage
```javascript
// Fetch signals
fetch('http://localhost:8000/api/v1/signals')
  .then(response => response.json())
  .then(data => {
    data.signals.forEach(signal => {
      console.log(`${signal.symbol}: ${signal.action} (${signal.confidence})`);
    });
  });

// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['signals'],
    symbols: ['AAPL', 'GOOGL']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### B. Configuration Examples

#### B.1 Docker Compose
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/goldensignals
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=goldensignals
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

#### B.2 Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: goldensignals-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: goldensignals-backend
  template:
    metadata:
      labels:
        app: goldensignals-backend
    spec:
      containers:
      - name: backend
        image: goldensignals:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: goldensignals-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

### C. Glossary

- **Agent**: An AI component that specializes in a specific aspect of market analysis
- **Confidence Score**: A measure (0-1) of how certain the AI is about a signal
- **RAG**: Retrieval-Augmented Generation, a technique for context-aware AI responses
- **Signal**: A recommendation to buy, sell, or hold a financial instrument
- **WebSocket**: A communication protocol for real-time data streaming

---

## PART VI: ADDITIONAL GUIDES

### 22. Quick Start Guide

#### 22.1 Get Started in 3 Steps

##### Prerequisites
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** (for cloning)

##### Choose Your Backend

**Option A: Quick Start (Recommended)**
```bash
# Start with optimized backend (real market data)
./start_dev.sh
```

**Option B: Backend Switcher**
```bash
# Make scripts executable
chmod +x start_dev.sh switch_backend.sh

# Start full backend with real market data
./switch_backend.sh full

# OR start mock backend for faster development
./switch_backend.sh mock

# Check current backend status
./switch_backend.sh status
```

##### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

#### 22.2 Backend Options

**Full Backend (Recommended for Production)**
- Real market data from yfinance
- Advanced signal generation with technical indicators
- WebSocket live updates
- Performance monitoring and caching
- Signal quality reporting
- Backtesting capabilities
- RSI, Bollinger Bands, SMA calculations
- Risk management with stop-loss/take-profit

**Mock Backend (Fast Development)**
- Fast startup (no external dependencies)
- Predictable data for testing
- All API endpoints mocked
- WebSocket support
- Perfect for frontend development

### 23. Deployment Guide

#### 23.1 Production Deployment

##### System Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv nginx supervisor postgresql redis-server

# Create application user
sudo useradd -m -s /bin/bash goldensignals
sudo su - goldensignals
```

##### Application Setup
```bash
# Clone repository
git clone https://github.com/yourusername/GoldenSignalsAI_V2.git
cd GoldenSignalsAI_V2

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
pip install gunicorn

# Build frontend
cd frontend
npm install --production
npm run build
cd ..
```

##### Gunicorn Configuration
Create `gunicorn_config.py`:
```python
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2
preload_app = True
accesslog = "/var/log/goldensignals/access.log"
errorlog = "/var/log/goldensignals/error.log"
loglevel = "info"
```

#### 23.2 Docker Deployment

##### Docker Setup
```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

##### Production Docker Compose
Use `docker-compose.prod.yml` for production deployment with proper scaling and monitoring.

#### 23.3 Kubernetes Deployment

##### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace goldensignals

# Apply configurations
kubectl apply -f k8s/production/ -n goldensignals

# Check deployment status
kubectl get pods -n goldensignals
kubectl get services -n goldensignals
```

### 24. Troubleshooting Guide

#### 24.1 Common Issues

##### Application Won't Start
**Symptoms:**
- Server crashes on startup
- Import errors
- Configuration errors

**Solutions:**
1. Check Python Version: `python --version` (Should be 3.9+)
2. Verify Virtual Environment: `source .venv/bin/activate`
3. Install Dependencies: `pip install -r requirements.txt`
4. Check Environment Variables: Verify `.env` file exists and is properly configured

##### Import Errors
**Symptoms:**
```python
ModuleNotFoundError: No module named 'src'
ImportError: cannot import name 'SignalGenerationEngine'
```

**Solutions:**
1. Fix Python Path: `export PYTHONPATH="${PYTHONPATH}:${PWD}"`
2. Install in Development Mode: `pip install -e .`
3. Check Module Structure: Verify `__init__.py` files exist

#### 24.2 API & Data Issues

##### yfinance HTTP 401 Errors
**Symptoms:**
```
ERROR:__main__:Error fetching market data for AAPL: HTTP Error 401
```

**Solutions:**
1. Use Direct yfinance API instead of fast_info
2. Enable Fallback Sources with API keys in `.env`
3. Clear yfinance Cache: `rm -rf ~/.cache/py-yfinance/`
4. Use Mock Data for Development: `MOCK_DATA_ENABLED=True`

##### No Market Data Returned
**Solutions:**
1. Check Market Hours: Verify market is open
2. Verify Symbol Validity: Test symbol directly with yfinance
3. Check Data Source Priority in rate_limit_handler.py

#### 24.3 Performance Issues

##### High Memory Usage
**Solutions:**
1. Limit Worker Processes in gunicorn_config.py
2. Enable Memory Profiling: `pip install memory-profiler`
3. Clear Caches Periodically with cleanup tasks

##### Slow API Response Times
**Solutions:**
1. Enable Redis Caching: `REDIS_URL=redis://localhost:6379/0`
2. Add Database Indexes for improved query performance
3. Use Batch Endpoints instead of multiple individual calls

### 25. Contributing Guide

#### 25.1 Development Setup
1. Fork the repository
2. Clone your fork
3. Create a virtual environment: `conda create -n goldensignalsai-dev python=3.10`
4. Install dependencies: `poetry install --no-root`

#### 25.2 Contribution Workflow
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `poetry run pytest --cov=./`
4. Run linters: `black .` and `flake8 .`
5. Commit your changes: `git commit -m "Description of changes"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

#### 25.3 Code Guidelines
- Follow PEP 8 style guide
- Write comprehensive unit tests
- Document new functions and classes
- Maintain modular architecture
- Use type hints

#### 25.4 Contribution Areas
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations
- Additional agent implementations

---

## PART VII: PROJECT STATUS & IMPLEMENTATION

### 26. Project Status Summary

#### 26.1 Current Version: 4.0 (Production Ready)

**✅ Completed Features:**
- Multi-agent consensus system with 30+ agents
- Real-time WebSocket data streaming
- AI-powered signal generation
- Professional trading UI with TradingView-style charts
- Comprehensive backtesting framework
- Enterprise monitoring and logging
- Horizontal scaling architecture
- Production CI/CD pipeline
- Component library with 50+ reusable components
- Complete API documentation
- Troubleshooting and deployment guides

**🚧 In Progress:**
- Advanced options strategies
- Multi-language support
- Mobile application
- Social trading features

**📅 Planned:**
- Automated trading execution
- Custom strategy builder
- Advanced portfolio optimization
- Institutional API

#### 26.2 Performance Metrics
- **Signal Generation**: <100ms average
- **WebSocket Latency**: <10ms
- **API Response Time**: <50ms (p95)
- **Test Coverage**: >80%
- **Uptime**: 99.9%

### 27. Implementation History

#### 27.1 Major Milestones
- **Phase 1**: Core architecture and backend implementation
- **Phase 2**: Frontend development and UI/UX design
- **Phase 3**: AI agent system and signal generation
- **Phase 4**: Real-time data integration and WebSocket implementation
- **Phase 5**: Production deployment and monitoring

#### 27.2 Technical Achievements
- Successfully implemented multi-agent consensus system
- Achieved sub-100ms signal generation response times
- Built production-ready scalable architecture
- Implemented comprehensive testing framework
- Created professional trading UI with dark theme
- Established CI/CD pipeline with automated testing

---

**End of Master Documentation**

*This comprehensive documentation consolidates all project information into a single, searchable resource. For the most up-to-date information, please refer to the individual source files if they still exist.*

*Last updated: December 2024* 