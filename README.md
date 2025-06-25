# GoldenSignalsAI V3 🚀

**Professional Signals-Only Trading Platform** - A sophisticated React/TypeScript frontend with FastAPI backend, designed for algorithmic trading signal analysis and visualization.

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20%2B%20Python-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### 🎯 **Signal Analysis Dashboard**
- **Real-time Signal Feed** - Live trading signals with confidence scores
- **Agent Performance Monitoring** - AI agent accuracy and performance metrics
- **Market Pulse** - Real-time market overview and sentiment analysis
- **Professional Charts** - TradingView-style charts with multiple timeframes

### 📊 **Advanced Charting**
- **Multiple Chart Types** - Candlestick, Line, and Bar charts
- **Price Projections** - ML-powered price movement predictions
- **Signal Overlays** - Buy/sell signals directly on charts
- **Interactive Controls** - Time period selection, symbol search, real-time updates

### 🤖 **AI-Powered Analytics**
- **Multi-Agent System** - Signal, Risk, and Market analysis agents
- **Confidence Scoring** - Each signal comes with AI confidence metrics
- **Historical Performance** - Track agent accuracy over time
- **Risk Assessment** - Intelligent risk analysis for each signal

### 🎨 **Professional UI/UX**
- **Dark Theme** - Easy on the eyes for long trading sessions
- **Responsive Design** - Works perfectly on desktop and tablet
- **TD Ameritrade Inspired** - Clean, professional trading interface
- **Real-time Updates** - Live data refresh without page reload

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for version control
- **PostgreSQL 12+** (optional, for persistence)
- **Redis 6+** (optional, for caching)

### 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GoldenSignalsAI_V2
   ```

2. **Start Everything** (Recommended)
   ```bash
   ./start.sh
   ```
   This single command will:
   - Check all prerequisites
   - Create virtual environment if needed
   - Install all dependencies
   - Start databases (if available)
   - Launch backend API
   - Launch frontend UI

3. **Alternative Commands**
   ```bash
   # Check service status
   ./start.sh status
   
   # Start only backend
   ./start.sh start --services backend
   
   # Start only frontend
   ./start.sh start --services frontend
   
   # View logs
   ./start.sh logs backend
   ./start.sh logs frontend
   
   # Stop all services
   ./start.sh stop
   
   # Get help
   ./start.sh help
   ```

4. **Access the application**
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

## 🔧 Development Tools

### **Master Startup Script**
All services are now managed through a single master script:

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

# Logs and debugging
./start.sh logs backend            # View backend logs
./start.sh logs frontend           # View frontend logs

# Maintenance
./start.sh install                 # Install/update dependencies
./start.sh help                    # Show all options
```

### **Legacy Scripts (Removed)**
The following scripts have been consolidated into `start.sh`:
- All `start_*.sh` scripts
- All `run_*.sh` scripts
- `restart-frontend.sh`
- `start-ui.sh`

For legacy compatibility, `dev-utils.sh` and `status-check.sh` remain available.

## 📁 Project Structure

```
GoldenSignalsAI_V2/
├── 📂 src/                          # Backend (FastAPI)
│   ├── main_simple.py               # Main FastAPI application
│   ├── 📂 services/                 # Business logic services
│   │   ├── market_data_service.py   # Market data provider
│   │   └── ml_model_service.py      # ML model inference
│   └── 📂 models/                   # Data models and schemas
│
├── 📂 frontend/                     # Frontend (React + TypeScript)
│   ├── 📂 src/
│   │   ├── 📂 pages/                # Main application pages
│   │   │   └── Dashboard/           # Trading dashboard
│   │   ├── 📂 components/           # Reusable components
│   │   │   ├── Chart/               # Trading chart components
│   │   │   └── Layout/              # Layout components
│   │   ├── 📂 services/             # API communication
│   │   └── 📂 store/                # State management (Zustand)
│   ├── package.json                 # Node.js dependencies
│   └── vite.config.ts               # Vite configuration
│
├── 📂 ml_training/                  # ML model training
│   └── 📂 models/                   # Trained model files
│
├── dev-utils.sh                     # Development utilities
├── status-check.sh                  # System status checker
└── README.md                        # This file
```

## 🔌 API Endpoints

### **Core Endpoints**
- `GET /health` - System health check
- `GET /api/v1/signals/{symbol}` - Get signals for symbol
- `GET /api/v1/signals/latest` - Get latest signals
- `GET /api/v1/market-data/{symbol}` - Get market data
- `GET /api/v1/agents/performance` - Get AI agent performance
- `GET /api/v1/market-summary` - Get market overview

### **Real-time Data**
- WebSocket connections for live signal updates
- Automatic data refresh every 30 seconds
- Historical data endpoints for charting

## 🎨 Frontend Architecture

### **Technology Stack**
- **React 18** with TypeScript
- **Material-UI (MUI)** for professional components
- **Lightweight Charts** for TradingView-style charts
- **TanStack Query** for data fetching and caching
- **Zustand** for state management
- **Vite** for fast development and building

### **Key Components**

#### **Dashboard** (`/src/pages/Dashboard/Dashboard.tsx`)
- Central command center for trading signals
- Real-time market data integration
- Agent performance monitoring
- Signal stream with live updates

#### **TradingChart** (`/src/components/Chart/TradingChart.tsx`)
- Professional trading charts with multiple types
- Real-time price data integration
- Signal overlay functionality
- Price projection visualization

#### **API Client** (`/src/services/api.ts`)
- Centralized API communication
- Error handling and retry logic
- TypeScript interfaces for type safety
- Real-time data subscriptions

## 🔧 Backend Architecture

### **Technology Stack**
- **FastAPI** for high-performance API
- **Python 3.8+** with asyncio support
- **Pandas & NumPy** for data processing
- **Scikit-learn** for ML model inference
- **CORS middleware** for frontend integration

### **Key Services**

#### **Market Data Service** (`/src/services/market_data_service.py`)
- Real-time market data generation
- Historical data simulation
- Price calculation algorithms
- Data caching and optimization

#### **ML Model Service** (`/src/services/ml_model_service.py`)
- AI model loading and inference
- Signal generation algorithms
- Confidence score calculation
- Performance tracking

## 🚦 Current Status

### ✅ **Completed Features**
- [x] Full frontend-backend integration
- [x] Real-time signal dashboard
- [x] Professional trading charts
- [x] AI agent performance monitoring
- [x] Market data visualization
- [x] Responsive UI design
- [x] API documentation
- [x] Development tools and utilities
- [x] Status monitoring systems

### 🔄 **Live Services**
- **Backend**: Running on port 8000
- **Frontend**: Running on port 3000
- **API**: Fully functional with real-time data
- **Charts**: Interactive with live price updates
- **Signals**: Real-time AI-generated trading signals

## 🎯 Usage Guide

### **Getting Started**
1. Open http://localhost:3000 in your browser
2. The dashboard loads with real-time market data
3. Use the symbol search to view different stocks
4. Toggle between chart types (Candlestick, Line, Bar)
5. Enable price projections to see AI predictions
6. Monitor signal feed for trading opportunities

### **Dashboard Overview**
- **Main Chart**: Central price chart with signal overlays
- **Signal Stream**: Live feed of AI-generated signals
- **Agent Performance**: Real-time agent accuracy metrics
- **Market Pulse**: Current market conditions overview

### **Chart Features**
- **Time Periods**: 1D, 5D, 1M, 3M, 6M, 1Y, 5Y
- **Chart Types**: Candlestick, Line, Bar charts
- **Price Projections**: Toggle ML-powered price predictions
- **Signal Markers**: Visual buy/sell signal indicators
- **Real-time Updates**: Live price and signal updates

## 🧪 Testing

### **Automated Testing**
```bash
# Test API endpoints
./dev-utils.sh test

# Check system status
./status-check.sh
```

### **Manual Testing**
1. Verify both services are running
2. Test symbol search functionality
3. Check real-time data updates
4. Validate chart interactions
5. Monitor signal generation

## 🤝 Contributing

### **Development Workflow**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### **Code Standards**
- **TypeScript** for frontend type safety
- **Python typing** for backend type hints
- **ESLint + Prettier** for code formatting
- **Comprehensive error handling**
- **Responsive design principles**

## 📚 Documentation

Visit our comprehensive [**Documentation Hub**](DOCUMENTATION_HUB.md) for organized access to all documentation, including:
- Complete implementation summaries for all 54 GitHub issues
- Architecture and design documentation
- Platform transformation reports
- Testing and deployment guides
- API documentation and troubleshooting

### **Quick Links**
- [Documentation Hub](DOCUMENTATION_HUB.md) - Central documentation index
- [Complete Implementation Summary](COMPLETE_IMPLEMENTATION_SUMMARY.md) - All 54 issues implemented
- [API Documentation](API_DOCUMENTATION.md) - REST API reference
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Local, Docker, Kubernetes, Cloud
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Common issues and solutions
- [Contributing](CONTRIBUTING.md) - How to contribute

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

### **Getting Help**
- Check the status with `./status-check.sh`
- View logs with `./dev-utils.sh logs [backend|frontend]`
- Test API with `./dev-utils.sh test`
- Clean environment with `./dev-utils.sh clean`

### **Common Issues**
- **Port conflicts**: Use `./dev-utils.sh clean` to kill existing processes
- **Dependencies**: Run `./dev-utils.sh setup` to reinstall dependencies
- **Data issues**: Restart services with `./dev-utils.sh restart`

---

**Built with ❤️ for professional traders and quantitative analysts**

*Ready for production deployment and real trading scenarios*
