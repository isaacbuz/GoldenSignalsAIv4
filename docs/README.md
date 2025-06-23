# GoldenSignalsAI Documentation Hub

Welcome to the comprehensive documentation for GoldenSignalsAI - an AI-powered trading signal platform.

## 📚 Documentation Structure

### 🚀 Getting Started
- [Main README](../README.md) - Project overview and quick start
- [Installation Guide](../SETUP_API_KEYS_GUIDE.md) - API keys and environment setup
- [Deployment Guide](../DEPLOYMENT_GUIDE.md) - Local, Docker, Kubernetes, and cloud deployment
- [Troubleshooting Guide](../TROUBLESHOOTING_GUIDE.md) - Common issues and solutions

### 🏗️ Architecture & Design
- [Project Structure](../PROJECT_STRUCTURE.md) - Codebase organization
- [Architecture Overview](../ARCHITECTURE_CONSOLIDATION_PLAN.md) - System design and components
- [API Documentation](../API_DOCUMENTATION.md) - REST API reference
- [Streamlined Architecture](../STREAMLINED_ARCHITECTURE.md) - Simplified system design

### 🤖 AI & Machine Learning
- [Agent Communication Architecture](../AGENT_COMMUNICATION_ARCHITECTURE.md) - Multi-agent system design
- [Adaptive Learning Architecture](../ADAPTIVE_LEARNING_ARCHITECTURE.md) - ML model adaptation
- [Advanced ML Models Guide](../ADVANCED_ML_MODELS_GUIDE.md) - ML implementation details
- [Hybrid Sentiment Architecture](../HYBRID_SENTIMENT_ARCHITECTURE.md) - Sentiment analysis system

### 📊 Trading Features
- [Signal Generation](../AGENT_SIGNALS_PERFORMANCE_GUIDE.md) - Signal generation and performance
- [Live Data Integration](../LIVE_DATA_INTEGRATION_GUIDE.md) - Real-time market data
- [Backtesting Guide](../BACKTESTING_ACCURACY_GUIDE.md) - Historical testing framework
- [Options Trading Roadmap](../PROJECT_REVIEW_OPTIONS_TRADING_ROADMAP.md) - Options strategy

### 🎨 Frontend & UI
- [Frontend Implementation Guide](../FRONTEND_IMPLEMENTATION_GUIDE.md) - React/TypeScript frontend
- [UI/UX Strategy](../UI_UX_STRATEGY.md) - Design principles
- [Chart Features](../VISUALIZATION_FEATURES_GUIDE.md) - Advanced charting capabilities
- [TradingView Integration](./frontend/src/components/AITradingLab/TRADINGVIEW_INTEGRATION_GUIDE.md)

### 🔧 Development
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Testing Guide](../TEST_RUNNER_GUIDE.md) - Running tests
- [CI/CD Implementation](../CI_CD_IMPLEMENTATION_GUIDE.md) - Automated workflows
- [Performance Optimization](../PERFORMANCE_OPTIMIZATION_GUIDE.md) - Speed improvements

### 📈 Roadmaps & Plans
- [Implementation Roadmap](../IMPLEMENTATION_ROADMAP.md) - Development timeline
- [Future Enhancements](../FUTURE_ENHANCEMENTS_ROADMAP.md) - Planned features
- [Production Training Roadmap](../PRODUCTION_TRAINING_ROADMAP.md) - ML model training
- [Comprehensive Improvement Plan](../COMPREHENSIVE_IMPROVEMENT_PLAN.md) - System enhancements

### 🛠️ Operations
- [Production Deployment](../PRODUCTION_ML_DEPLOYMENT_GUIDE.md) - Production ML setup
- [Data Sources Guide](../DATA_SOURCES_GUIDE.md) - Market data providers
- [Database Setup](../DATABASE_SETUP_GUIDE.md) - PostgreSQL configuration
- [Rate Limit Solutions](../RATE_LIMIT_SOLUTIONS.md) - API rate limiting

### 🎯 Specific Features
- [AI Signal Prophet](../AI_SIGNAL_PROPHET_GUIDE.md) - AI-powered signal generation
- [AI Trading Lab](../AI_TRADING_LAB_IMPLEMENTATION.md) - Advanced trading features
- [Pattern Recognition](../PATTERN_RECOGNITION_ENHANCEMENT_PLAN.md) - Chart patterns
- [MCP Architecture](../MCP_ARCHITECTURE_DESIGN.md) - Microservices design

## 🗂️ Module-Specific Documentation

### Agents Module
- [Agents README](../agents/README.md) - Agent system overview
- [ML Research](../agents/research/ml/README.md) - ML research agents
- [Strategy Agents](../agents/strategy/README.md) - Trading strategy agents
- [Technical Analysis](../agents/core/technical/README.md) - Technical indicators
- [Sentiment Analysis](../agents/core/sentiment/README.md) - Market sentiment

### Frontend Module
- [Frontend README](../frontend/README.md) - React application
- [UI Architecture](../frontend/ENHANCED_UI_ARCHITECTURE.md) - Component structure
- [UI Improvements Guide](../frontend/UI_IMPROVEMENTS_GUIDE.md) - UI enhancements

### ML Training Module
- [ML Research](../src/ml/research/README.md) - ML research documentation
- [Live Evaluation](../ml_training/README_LIVE_EVALUATION.md) - Model evaluation

## 📋 Quick Reference

### Essential Commands
```bash
# Start all services
./start.sh

# Run tests
./run_tests.sh

# Check system status
./status-check.sh

# Deploy to production
./deploy_production.sh
```

### Key API Endpoints
- `GET /api/v1/signals/latest` - Latest trading signals
- `GET /api/v1/market-data/{symbol}` - Market data
- `GET /api/v1/agents/performance` - Agent performance
- `WebSocket /ws` - Real-time updates

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API access
- `ALPHA_VANTAGE_API_KEY` - Market data
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis cache

## 🔍 Finding Documentation

### By Feature
- **Trading Signals**: See Signal Generation guides
- **Live Data**: Check Live Data Integration
- **Backtesting**: Review Backtesting Guide
- **AI Features**: Explore AI & Machine Learning section

### By Role
- **Developers**: Start with Architecture & Development sections
- **Traders**: Focus on Trading Features and UI guides
- **DevOps**: Check Operations and Deployment guides
- **Contributors**: Read Contributing Guide first

## 📊 Project Status

### Completed Features ✅
- Multi-agent trading system
- Real-time signal generation
- Professional trading UI
- Comprehensive backtesting
- Live market data integration
- AI-powered analysis

### In Progress 🚧
- Options trading (Issue #8)
- Advanced ML models
- Performance optimizations
- Additional data sources

### GitHub Issues
- View [Open Issues](https://github.com/isaacbuz/GoldenSignalsAIv4/issues)
- Check [Closed Issues](https://github.com/isaacbuz/GoldenSignalsAIv4/issues?q=is%3Aissue+is%3Aclosed) for completed work

## 🆘 Getting Help

1. Check the [Troubleshooting Guide](../TROUBLESHOOTING_GUIDE.md)
2. Review relevant documentation sections
3. Search [GitHub Issues](https://github.com/isaacbuz/GoldenSignalsAIv4/issues)
4. Create a new issue if needed

---

*Last updated: June 2024* 