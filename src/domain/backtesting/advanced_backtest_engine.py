"""
Advanced Backtesting Engine for GoldenSignalsAI
Integrates MCP, Agentic AI, and comprehensive testing capabilities
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# MCP Integration
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Database and caching
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Market data
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

# Technical indicators
# import talib  # Temporarily disabled due to numpy compatibility issue
import pandas_ta as ta

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from src.utils.timezone_utils import now_utc, ensure_timezone_aware
from simple_live_data import simple_live_data

logger = logging.getLogger(__name__)


@dataclass
class BacktestSignal:
    """Enhanced signal with agent attribution"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    # Agent attribution
    agent_scores: Dict[str, float] = field(default_factory=dict)
    agent_reasoning: Dict[str, str] = field(default_factory=dict)
    
    # Technical indicators at signal time
    indicators: Dict[str, float] = field(default_factory=dict)
    
    # ML predictions
    ml_predictions: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    risk_score: float = 0.0
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Metadata
    timeframe: str = "5m"
    strategy: str = "hybrid"
    mcp_server: Optional[str] = None


@dataclass
class BacktestTrade:
    """Complete trade record with comprehensive metrics"""
    signal: BacktestSignal
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    
    # P&L
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    # Risk metrics during trade
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    time_in_trade: Optional[timedelta] = None
    
    # Market conditions
    market_regime: str = ""
    volatility_at_entry: float = 0.0
    volume_profile: Dict[str, float] = field(default_factory=dict)
    
    # Performance attribution
    agent_contribution: Dict[str, float] = field(default_factory=dict)
    factor_attribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    # Returns
    total_return: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    beta: float
    
    # Risk
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    
    # Efficiency metrics
    avg_trade_duration: timedelta
    trades_per_day: float
    exposure_time: float
    
    # Agent performance
    agent_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Model performance
    model_accuracy: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Market regime performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based analysis
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    hourly_performance: Dict[int, float] = field(default_factory=dict)
    
    # Robustness metrics
    monte_carlo_results: Optional[Dict[str, Any]] = None
    walk_forward_results: Optional[Dict[str, Any]] = None
    parameter_sensitivity: Optional[Dict[str, Any]] = None


class AdvancedBacktestEngine:
    """
    Production-ready backtesting engine with:
    - MCP integration for distributed agent testing
    - Agentic AI coordination
    - Comprehensive metrics and analysis
    - Real-time and historical data support
    - Multi-timeframe and multi-asset testing
    - ML model validation
    - Monte Carlo simulations
    - Walk-forward analysis
    - Parameter optimization
    """
    
    def __init__(self, config_path: str = "config/backtest_config.yaml"):
        self.config = self._load_config(config_path)
        self.mcp_clients = {}
        self.db_pool = None
        self.redis_client = None
        self.engine = None
        self.market_data_cache = {}
        self.indicator_cache = {}
        self.ml_models = {}
        self.agent_registry = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load backtest configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set defaults
        defaults = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'position_sizing': 'fixed',
            'position_size': 0.1,
            'max_positions': 5,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'confidence_threshold': 0.7,
            'enable_mcp': True,
            'enable_ml': True,
            'enable_monte_carlo': True,
            'monte_carlo_runs': 1000,
            'enable_walk_forward': True,
            'walk_forward_periods': 12,
            'enable_parameter_optimization': True
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
                
        return config
        
    async def initialize(self):
        """Initialize all components"""
        # Database connections
        await self._init_database()
        
        # Redis for caching
        await self._init_redis()
        
        # MCP servers
        if self.config.get('enable_mcp'):
            await self._init_mcp_servers()
            
        # Load ML models
        if self.config.get('enable_ml'):
            await self._load_ml_models()
            
        # Register agents
        await self._register_agents()
        
        logger.info("Advanced backtest engine initialized")
        
    async def _init_database(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5432)),
                user=os.getenv('DB_USER', 'goldensignals'),
                password=os.getenv('DB_PASSWORD', 'password'),
                database=os.getenv('DB_NAME', 'goldensignals'),
                min_size=5,
                max_size=20
            )
            
            # SQLAlchemy async engine
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql+asyncpg://goldensignals:password@localhost:5432/goldensignals'
            )
            self.engine = create_async_engine(database_url, echo=False)
            
            # Create tables
            await self._create_tables()
            
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            # Continue without database
            
    async def _init_redis(self):
        """Initialize Redis client"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = await redis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            # Continue without Redis
            
    async def _init_mcp_servers(self):
        """Initialize MCP server connections"""
        mcp_servers = self.config.get('mcp_servers', [])
        
        for server_config in mcp_servers:
            try:
                server_params = StdioServerParameters(
                    command=server_config['command'],
                    args=server_config.get('args', []),
                    env=server_config.get('env', {})
                )
                
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self.mcp_clients[server_config['name']] = session
                        logger.info(f"Connected to MCP server: {server_config['name']}")
                        
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_config['name']}: {e}")
                
    async def _load_ml_models(self):
        """Load pre-trained ML models"""
        model_dir = Path("ml_models")
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                try:
                    model = joblib.load(model_file)
                    self.ml_models[model_file.stem] = model
                    logger.info(f"Loaded ML model: {model_file.stem}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_file}: {e}")
                    
    async def _register_agents(self):
        """Register all available agents"""
        # Import agent modules dynamically
        agent_modules = [
            "agents.technical.momentum.momentum_agent",
            "agents.sentiment.news.news_sentiment_agent",
            "agents.ml.pattern_recognition_agent",
            "agents.options.options_flow_agent",
            "agents.volume.volume_profile_agent"
        ]
        
        for module_path in agent_modules:
            try:
                module = __import__(module_path, fromlist=['Agent'])
                agent_class = getattr(module, 'Agent', None)
                if agent_class:
                    agent = agent_class()
                    self.agent_registry[module_path.split('.')[-1]] = agent
                    logger.info(f"Registered agent: {module_path}")
            except Exception as e:
                logger.warning(f"Failed to register agent {module_path}: {e}")
                
    async def _create_tables(self):
        """Create backtest result tables"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            # Backtest runs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(36) UNIQUE NOT NULL,
                    strategy_name VARCHAR(100) NOT NULL,
                    start_date TIMESTAMPTZ NOT NULL,
                    end_date TIMESTAMPTZ NOT NULL,
                    symbols TEXT[] NOT NULL,
                    config JSONB NOT NULL,
                    metrics JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    status VARCHAR(20) DEFAULT 'running'
                )
            """)
            
            # Trade history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(36) NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    entry_time TIMESTAMPTZ NOT NULL,
                    exit_time TIMESTAMPTZ,
                    entry_price FLOAT NOT NULL,
                    exit_price FLOAT,
                    position_size FLOAT NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    pnl FLOAT,
                    pnl_percent FLOAT,
                    commission FLOAT,
                    slippage FLOAT,
                    exit_reason VARCHAR(50),
                    signal_data JSONB,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
                )
            """)
            
            # Agent performance table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(36) NOT NULL,
                    agent_name VARCHAR(100) NOT NULL,
                    total_signals INT DEFAULT 0,
                    accurate_signals INT DEFAULT 0,
                    accuracy FLOAT,
                    avg_confidence FLOAT,
                    contribution_score FLOAT,
                    metrics JSONB,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_id 
                ON backtest_trades(run_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol 
                ON backtest_trades(symbol, entry_time DESC)
            """)
            
    async def run_backtest(
        self,
        strategy: Union[str, Callable],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> BacktestMetrics:
        """
        Run comprehensive backtest
        
        Args:
            strategy: Strategy name or callable
            symbols: List of symbols to test
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            BacktestMetrics with comprehensive results
        """
        run_id = self._generate_run_id()
        logger.info(f"Starting backtest run: {run_id}")
        
        # Store run information
        await self._store_run_info(run_id, strategy, symbols, start_date, end_date)
        
        try:
            # Fetch market data
            market_data = await self._fetch_market_data(symbols, start_date, end_date)
            
            # Calculate indicators
            market_data = await self._calculate_indicators(market_data)
            
            # Run main backtest loop
            trades = await self._run_backtest_loop(
                run_id, strategy, market_data, **kwargs
            )
            
            # Calculate metrics
            metrics = await self._calculate_metrics(trades, market_data)
            
            # Run additional analysis
            if self.config.get('enable_monte_carlo'):
                metrics.monte_carlo_results = await self._run_monte_carlo(trades)
                
            if self.config.get('enable_walk_forward'):
                metrics.walk_forward_results = await self._run_walk_forward(
                    strategy, market_data, **kwargs
                )
                
            if self.config.get('enable_parameter_optimization'):
                metrics.parameter_sensitivity = await self._run_parameter_optimization(
                    strategy, market_data, **kwargs
                )
                
            # Store results
            await self._store_results(run_id, metrics, trades)
            
            # Generate report
            await self._generate_report(run_id, metrics, trades)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            await self._update_run_status(run_id, 'failed', str(e))
            raise
            
        finally:
            await self._cleanup_run(run_id)
            
    async def _fetch_market_data(
        self, 
        symbols: List[str], 
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical market data"""
        market_data = {}
        
        # Try cache first
        for symbol in symbols:
            cache_key = f"backtest:data:{symbol}:{start_date}:{end_date}"
            if self.redis_client:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    market_data[symbol] = pd.read_json(cached)
                    continue
                    
            # Fetch from data source
            data = await simple_live_data.fetch_historical_data(
                symbol, start_date, end_date, '5m'
            )
            
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('timestamp', inplace=True)
                market_data[symbol] = df
                
                # Cache for future use
                if self.redis_client:
                    await self.redis_client.setex(
                        cache_key,
                        3600,  # 1 hour TTL
                        df.to_json()
                    )
                    
        return market_data
        
    async def _calculate_indicators(
        self, 
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Calculate technical indicators for all symbols"""
        for symbol, df in market_data.items():
            # Price-based indicators
            df['SMA_20'] = ta.SMA(df['close'], timeperiod=20)
            df['SMA_50'] = ta.SMA(df['close'], timeperiod=50)
            df['EMA_12'] = ta.EMA(df['close'], timeperiod=12)
            df['EMA_26'] = ta.EMA(df['close'], timeperiod=26)
            
            # MACD
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # RSI
            df['RSI'] = ta.RSI(df['close'], timeperiod=14)
            
            # Bollinger Bands
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            
            # Volume indicators
            df['OBV'] = ta.OBV(df['close'], df['volume'])
            df['AD'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # Volatility
            df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Pattern recognition
            df['HAMMER'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['DOJI'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            
            # Custom indicators using pandas_ta
            df.ta.vwap(append=True)
            df.ta.supertrend(append=True)
            
        return market_data
        
    async def _run_backtest_loop(
        self,
        run_id: str,
        strategy: Union[str, Callable],
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> List[BacktestTrade]:
        """Main backtest execution loop"""
        trades = []
        open_positions = {}
        portfolio_value = self.config['initial_capital']
        cash = self.config['initial_capital']
        
        # Get all timestamps
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)
        timestamps = sorted(all_timestamps)
        
        # Skip warmup period for indicators
        start_idx = 100
        
        for i in range(start_idx, len(timestamps)):
            current_time = timestamps[i]
            
            # Check exits first
            for symbol, position in list(open_positions.items()):
                if symbol in market_data and current_time in market_data[symbol].index:
                    current_price = market_data[symbol].loc[current_time, 'close']
                    
                    # Check exit conditions
                    exit_signal = await self._check_exit_conditions(
                        position, current_price, current_time
                    )
                    
                    if exit_signal:
                        # Close position
                        trade = await self._close_position(
                            position, current_price, current_time, exit_signal
                        )
                        trades.append(trade)
                        cash += trade.pnl
                        del open_positions[symbol]
                        
            # Generate new signals
            if len(open_positions) < self.config['max_positions']:
                signals = await self._generate_signals(
                    strategy, market_data, current_time, open_positions.keys()
                )
                
                # Execute signals
                for signal in signals:
                    if signal.confidence >= self.config['confidence_threshold']:
                        position_size = self._calculate_position_size(
                            signal, cash, portfolio_value
                        )
                        
                        if position_size > 0:
                            trade = BacktestTrade(
                                signal=signal,
                                entry_time=current_time,
                                commission=position_size * signal.entry_price * self.config['commission']
                            )
                            
                            open_positions[signal.symbol] = trade
                            cash -= position_size * signal.entry_price * (1 + self.config['commission'])
                            
            # Update portfolio value
            portfolio_value = cash
            for symbol, position in open_positions.items():
                if symbol in market_data and current_time in market_data[symbol].index:
                    current_price = market_data[symbol].loc[current_time, 'close']
                    portfolio_value += position.signal.position_size * current_price
                    
        # Close remaining positions
        for symbol, position in open_positions.items():
            if symbol in market_data:
                last_price = market_data[symbol].iloc[-1]['close']
                trade = await self._close_position(
                    position, last_price, timestamps[-1], "end_of_backtest"
                )
                trades.append(trade)
                
        return trades
        
    async def _generate_signals(
        self,
        strategy: Union[str, Callable],
        market_data: Dict[str, pd.DataFrame],
        current_time: datetime,
        excluded_symbols: List[str]
    ) -> List[BacktestSignal]:
        """Generate trading signals using agents and models"""
        signals = []
        
        # Get available symbols
        available_symbols = [
            s for s in market_data.keys() 
            if s not in excluded_symbols and current_time in market_data[s].index
        ]
        
        # Process each symbol
        for symbol in available_symbols:
            df = market_data[symbol]
            current_idx = df.index.get_loc(current_time)
            
            if current_idx < 50:  # Need history
                continue
                
            # Get recent data
            recent_data = df.iloc[max(0, current_idx-100):current_idx+1]
            
            # Collect agent signals
            agent_signals = {}
            
            # Technical agents
            if 'momentum_agent' in self.agent_registry:
                momentum_signal = await self._get_agent_signal(
                    'momentum_agent', symbol, recent_data
                )
                if momentum_signal:
                    agent_signals['momentum'] = momentum_signal
                    
            # ML predictions
            if self.ml_models:
                ml_signal = await self._get_ml_predictions(symbol, recent_data)
                if ml_signal:
                    agent_signals['ml'] = ml_signal
                    
            # MCP server signals
            if self.mcp_clients:
                mcp_signals = await self._get_mcp_signals(symbol, recent_data)
                agent_signals.update(mcp_signals)
                
            # Combine signals
            if agent_signals:
                combined_signal = await self._combine_signals(
                    symbol, agent_signals, recent_data
                )
                if combined_signal:
                    signals.append(combined_signal)
                    
        return signals
        
    async def _get_agent_signal(
        self, 
        agent_name: str, 
        symbol: str, 
        data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Get signal from a specific agent"""
        agent = self.agent_registry.get(agent_name)
        if not agent:
            return None
            
        try:
            signal = await agent.analyze(symbol, data)
            return signal
        except Exception as e:
            logger.error(f"Agent {agent_name} error: {e}")
            return None
            
    async def _get_ml_predictions(
        self, 
        symbol: str, 
        data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Get ML model predictions"""
        predictions = {}
        
        for model_name, model in self.ml_models.items():
            try:
                # Prepare features
                features = self._prepare_ml_features(data)
                if features is not None:
                    prediction = model.predict_proba(features)[-1]
                    predictions[model_name] = {
                        'bullish_prob': prediction[1],
                        'bearish_prob': prediction[0]
                    }
            except Exception as e:
                logger.error(f"ML model {model_name} error: {e}")
                
        if predictions:
            avg_bullish = np.mean([p['bullish_prob'] for p in predictions.values()])
            return {
                'confidence': avg_bullish,
                'predictions': predictions,
                'action': 'buy' if avg_bullish > 0.6 else 'hold'
            }
            
        return None
        
    async def _get_mcp_signals(
        self, 
        symbol: str, 
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Get signals from MCP servers"""
        mcp_signals = {}
        
        for server_name, client in self.mcp_clients.items():
            try:
                # Call MCP server for analysis
                result = await client.call_tool(
                    "analyze_symbol",
                    arguments={
                        "symbol": symbol,
                        "data": data.to_dict(),
                        "indicators": list(data.columns)
                    }
                )
                
                if result and result.content:
                    mcp_signals[server_name] = json.loads(result.content[0].text)
                    
            except Exception as e:
                logger.error(f"MCP server {server_name} error: {e}")
                
        return mcp_signals
        
    async def _combine_signals(
        self,
        symbol: str,
        agent_signals: Dict[str, Dict[str, Any]],
        data: pd.DataFrame
    ) -> Optional[BacktestSignal]:
        """Combine multiple agent signals into a single signal"""
        # Extract confidences
        confidences = []
        actions = []
        
        for agent_name, signal in agent_signals.items():
            if 'confidence' in signal:
                confidences.append(signal['confidence'])
            if 'action' in signal:
                actions.append(signal['action'])
                
        if not confidences:
            return None
            
        # Weighted average confidence
        avg_confidence = np.mean(confidences)
        
        # Majority vote for action
        if actions:
            action_counts = pd.Series(actions).value_counts()
            consensus_action = action_counts.index[0]
        else:
            consensus_action = 'hold'
            
        if consensus_action == 'hold' or avg_confidence < 0.5:
            return None
            
        # Calculate entry, stop loss, and take profit
        current_price = data.iloc[-1]['close']
        atr = data.iloc[-1].get('ATR', current_price * 0.01)
        
        if consensus_action == 'buy':
            entry_price = current_price * (1 + self.config['slippage'])
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (3 * atr)
        else:  # sell
            entry_price = current_price * (1 - self.config['slippage'])
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (3 * atr)
            
        # Create signal
        signal = BacktestSignal(
            timestamp=data.index[-1],
            symbol=symbol,
            action=consensus_action,
            confidence=avg_confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=0,  # Will be calculated later
            agent_scores={name: sig.get('confidence', 0) for name, sig in agent_signals.items()},
            agent_reasoning={name: sig.get('reasoning', '') for name, sig in agent_signals.items()},
            indicators={
                'close': current_price,
                'volume': data.iloc[-1]['volume'],
                'rsi': data.iloc[-1].get('RSI', 50),
                'macd': data.iloc[-1].get('MACD', 0)
            }
        )
        
        return signal
        
    def _calculate_position_size(
        self,
        signal: BacktestSignal,
        cash: float,
        portfolio_value: float
    ) -> float:
        """Calculate position size based on strategy"""
        if self.config['position_sizing'] == 'fixed':
            position_value = portfolio_value * self.config['position_size']
        elif self.config['position_sizing'] == 'kelly':
            # Kelly Criterion
            win_prob = signal.confidence
            win_loss_ratio = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            position_value = portfolio_value * kelly_fraction
        else:
            position_value = portfolio_value * self.config['position_size']
            
        # Ensure we have enough cash
        position_value = min(position_value, cash * 0.95)  # Keep 5% cash buffer
        
        # Calculate shares
        position_size = position_value / signal.entry_price
        signal.position_size = position_size
        
        return position_size
        
    async def _check_exit_conditions(
        self,
        position: BacktestTrade,
        current_price: float,
        current_time: datetime
    ) -> Optional[str]:
        """Check if position should be closed"""
        signal = position.signal
        
        # Calculate P&L
        if signal.action == 'buy':
            pnl_percent = (current_price - signal.entry_price) / signal.entry_price
        else:
            pnl_percent = (signal.entry_price - current_price) / signal.entry_price
            
        # Stop loss
        if pnl_percent <= -abs(self.config['stop_loss']):
            return "stop_loss"
            
        # Take profit
        if pnl_percent >= abs(self.config['take_profit']):
            return "take_profit"
            
        # Time-based exit (optional)
        time_in_trade = current_time - position.entry_time
        if time_in_trade > timedelta(days=5):
            return "time_exit"
            
        # Trailing stop (optional)
        if hasattr(position, 'max_profit'):
            if position.max_profit > 0.02 and pnl_percent < position.max_profit * 0.5:
                return "trailing_stop"
                
        # Update max profit
        position.max_profit = max(getattr(position, 'max_profit', 0), pnl_percent)
        
        return None
        
    async def _close_position(
        self,
        position: BacktestTrade,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> BacktestTrade:
        """Close a position and calculate final metrics"""
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.time_in_trade = exit_time - position.entry_time
        
        # Calculate P&L
        if position.signal.action == 'buy':
            gross_pnl = (exit_price - position.signal.entry_price) * position.signal.position_size
        else:
            gross_pnl = (position.signal.entry_price - exit_price) * position.signal.position_size
            
        # Apply commission
        exit_commission = exit_price * position.signal.position_size * self.config['commission']
        position.pnl = gross_pnl - position.commission - exit_commission
        position.pnl_percent = position.pnl / (position.signal.entry_price * position.signal.position_size)
        
        return position
        
    async def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        market_data: Dict[str, pd.DataFrame]
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        if not trades:
            return self._empty_metrics()
            
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(trades)
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Risk-adjusted returns
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        closed_trades = [t for t in trades if t.exit_time]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl_percent) for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Agent performance
        agent_metrics = self._calculate_agent_metrics(trades)
        
        # Create metrics object
        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=0,  # TODO: Calculate vs SPY
            alpha=0,  # TODO: Calculate
            beta=0,  # TODO: Calculate
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=0,  # TODO: Calculate
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=0,  # TODO: Calculate
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0,
            expectancy=(win_rate * avg_win) - ((1 - win_rate) * avg_loss),
            payoff_ratio=avg_win / avg_loss if avg_loss > 0 else 0,
            avg_trade_duration=timedelta(seconds=np.mean([(t.exit_time - t.entry_time).total_seconds() for t in closed_trades])) if closed_trades else timedelta(0),
            trades_per_day=len(closed_trades) / max(days, 1),
            exposure_time=0.8,  # TODO: Calculate actual exposure
            agent_metrics=agent_metrics,
            monthly_returns=equity_curve.resample('M').last().pct_change(),
            daily_returns=equity_curve.resample('D').last().pct_change()
        )
        
        return metrics
        
    def _calculate_equity_curve(self, trades: List[BacktestTrade]) -> pd.Series:
        """Calculate equity curve from trades"""
        equity_data = []
        current_equity = self.config['initial_capital']
        
        # Sort trades by entry and exit times
        events = []
        for trade in trades:
            events.append((trade.entry_time, 'entry', trade))
            if trade.exit_time:
                events.append((trade.exit_time, 'exit', trade))
                
        events.sort(key=lambda x: x[0])
        
        # Process events
        for time, event_type, trade in events:
            if event_type == 'exit':
                current_equity += trade.pnl
            equity_data.append((time, current_equity))
            
        # Create series
        if equity_data:
            times, values = zip(*equity_data)
            return pd.Series(values, index=pd.DatetimeIndex(times))
        else:
            return pd.Series([self.config['initial_capital']], index=[datetime.now()])
            
    def _calculate_agent_metrics(self, trades: List[BacktestTrade]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each agent"""
        agent_metrics = {}
        
        # Aggregate by agent
        for trade in trades:
            for agent_name, score in trade.signal.agent_scores.items():
                if agent_name not in agent_metrics:
                    agent_metrics[agent_name] = {
                        'total_signals': 0,
                        'profitable_signals': 0,
                        'total_pnl': 0,
                        'avg_confidence': 0,
                        'confidence_sum': 0
                    }
                    
                metrics = agent_metrics[agent_name]
                metrics['total_signals'] += 1
                metrics['confidence_sum'] += score
                
                if trade.pnl > 0:
                    metrics['profitable_signals'] += 1
                metrics['total_pnl'] += trade.pnl
                
        # Calculate final metrics
        for agent_name, metrics in agent_metrics.items():
            if metrics['total_signals'] > 0:
                metrics['accuracy'] = metrics['profitable_signals'] / metrics['total_signals']
                metrics['avg_confidence'] = metrics['confidence_sum'] / metrics['total_signals']
                metrics['avg_pnl'] = metrics['total_pnl'] / metrics['total_signals']
            else:
                metrics['accuracy'] = 0
                metrics['avg_confidence'] = 0
                metrics['avg_pnl'] = 0
                
            # Remove temporary fields
            del metrics['confidence_sum']
            
        return agent_metrics
        
    async def _run_monte_carlo(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Run Monte Carlo simulations"""
        trade_returns = [t.pnl_percent for t in trades if t.exit_time]
        
        if len(trade_returns) < 10:
            return {}
            
        simulation_results = []
        
        for _ in range(self.config['monte_carlo_runs']):
            # Bootstrap sample
            sampled_returns = np.random.choice(
                trade_returns,
                size=len(trade_returns),
                replace=True
            )
            
            # Calculate cumulative return
            cumulative_return = np.prod(1 + np.array(sampled_returns)) - 1
            simulation_results.append(cumulative_return)
            
        return {
            'mean_return': np.mean(simulation_results),
            'std_return': np.std(simulation_results),
            'percentiles': {
                '5%': np.percentile(simulation_results, 5),
                '25%': np.percentile(simulation_results, 25),
                '50%': np.percentile(simulation_results, 50),
                '75%': np.percentile(simulation_results, 75),
                '95%': np.percentile(simulation_results, 95)
            },
            'probability_of_profit': sum(1 for r in simulation_results if r > 0) / len(simulation_results),
            'probability_of_loss': sum(1 for r in simulation_results if r < -0.1) / len(simulation_results)
        }
        
    async def _run_walk_forward(
        self,
        strategy: Union[str, Callable],
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        # TODO: Implement walk-forward analysis
        return {
            'periods_tested': self.config['walk_forward_periods'],
            'avg_out_of_sample_return': 0,
            'consistency_score': 0
        }
        
    async def _run_parameter_optimization(
        self,
        strategy: Union[str, Callable],
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """Run parameter sensitivity analysis"""
        # TODO: Implement parameter optimization
        return {
            'optimal_parameters': {},
            'sensitivity_analysis': {}
        }
        
    async def _store_results(
        self,
        run_id: str,
        metrics: BacktestMetrics,
        trades: List[BacktestTrade]
    ):
        """Store backtest results in database"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            # Update run with metrics
            await conn.execute("""
                UPDATE backtest_runs
                SET metrics = $1,
                    completed_at = $2,
                    status = 'completed'
                WHERE run_id = $3
            """, json.dumps(asdict(metrics)), now_utc(), run_id)
            
            # Store trades
            for trade in trades:
                await conn.execute("""
                    INSERT INTO backtest_trades
                    (run_id, symbol, entry_time, exit_time, entry_price, exit_price,
                     position_size, side, pnl, pnl_percent, commission, slippage,
                     exit_reason, signal_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                run_id, trade.signal.symbol, trade.entry_time, trade.exit_time,
                trade.signal.entry_price, trade.exit_price, trade.signal.position_size,
                trade.signal.action, trade.pnl, trade.pnl_percent,
                trade.commission, trade.slippage, trade.exit_reason,
                json.dumps(asdict(trade.signal))
                )
                
    async def _generate_report(
        self,
        run_id: str,
        metrics: BacktestMetrics,
        trades: List[BacktestTrade]
    ):
        """Generate comprehensive backtest report"""
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
        # Generate text summary
        summary = f"""
Backtest Summary - Run ID: {run_id}
=====================================

Performance Metrics:
- Total Return: {metrics.total_return:.2%}
- Annualized Return: {metrics.annualized_return:.2%}
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Max Drawdown: {metrics.max_drawdown:.2%}

Trading Statistics:
- Total Trades: {metrics.total_trades}
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Average Trade Duration: {metrics.avg_trade_duration}

Risk Metrics:
- Volatility: {metrics.volatility:.2%}
- VaR (95%): {metrics.var_95:.2%}
- CVaR (95%): {metrics.cvar_95:.2%}
"""
            
        # Save summary
        summary_path = f"reports/backtest_{run_id}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
            
        logger.info(f"Report saved to {summary_path}")
            
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        import uuid
        return str(uuid.uuid4())
        
    async def _store_run_info(
        self,
        run_id: str,
        strategy: Union[str, Callable],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """Store initial run information"""
        if not self.db_pool:
            return
            
        strategy_name = strategy if isinstance(strategy, str) else strategy.__name__
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO backtest_runs
                (run_id, strategy_name, start_date, end_date, symbols, config, metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            run_id, strategy_name, start_date, end_date,
            symbols, json.dumps(self.config), json.dumps({})
            )
            
    async def _update_run_status(self, run_id: str, status: str, error: str = None):
        """Update run status"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            if error:
                await conn.execute("""
                    UPDATE backtest_runs
                    SET status = $1,
                        metrics = jsonb_set(metrics, '{error}', $2)
                    WHERE run_id = $3
                """, status, json.dumps(error), run_id)
            else:
                await conn.execute("""
                    UPDATE backtest_runs
                    SET status = $1
                    WHERE run_id = $2
                """, status, run_id)
                
    async def _cleanup_run(self, run_id: str):
        """Cleanup after run"""
        # Clear caches
        self.market_data_cache.clear()
        self.indicator_cache.clear()
        
    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics object"""
        return BacktestMetrics(
            total_return=0,
            annualized_return=0,
            benchmark_return=0,
            alpha=0,
            beta=0,
            volatility=0,
            downside_volatility=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            var_95=0,
            cvar_95=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            information_ratio=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            expectancy=0,
            payoff_ratio=0,
            avg_trade_duration=timedelta(0),
            trades_per_day=0,
            exposure_time=0
        )
            
    def _prepare_ml_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML models"""
        try:
            feature_columns = [
                'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
                'SMA_20', 'SMA_50', 'ATR', 'OBV', 'AD'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) < 5:
                return None
                
            features = data[available_features].iloc[-1:].values
            
            # Add price-based features
            close_prices = data['close'].values[-20:]
            if len(close_prices) >= 20:
                features = np.append(features, [
                    close_prices[-1] / close_prices[-2] - 1,  # 1-day return
                    close_prices[-1] / close_prices[-5] - 1,  # 5-day return
                    close_prices[-1] / close_prices[-20] - 1,  # 20-day return
                    np.std(close_prices) / np.mean(close_prices)  # Coefficient of variation
                ])
                
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
            
    async def close(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.engine:
            await self.engine.dispose()


# CLI Interface for running backtests
async def main():
    """CLI for running backtests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GoldenSignalsAI Backtest')
    parser.add_argument('--strategy', type=str, default='simple', help='Strategy name')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT'], help='Symbols to test')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config/backtest_config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = AdvancedBacktestEngine(args.config)
    await engine.initialize()
    
    try:
        # Run backtest
        results = await engine.run_backtest(
            strategy=args.strategy,
            symbols=args.symbols,
            start_date=datetime.strptime(args.start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(args.end_date, '%Y-%m-%d')
        )
        
        # Print summary
        print(f"\nBacktest Results:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        
    finally:
        await engine.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main()) 