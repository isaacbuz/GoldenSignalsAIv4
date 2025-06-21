"""
Comprehensive Backtesting Engine for GoldenSignalsAI
Combines the best features from all backtest implementations
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Unified configuration for backtesting"""
    # Basic settings
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    
    # Position sizing
    position_size: float = 0.1  # Fraction of capital per trade
    max_positions: int = 5
    position_sizing_method: str = "fixed"  # 'fixed', 'kelly', 'risk_parity'
    
    # Trading costs
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    # Risk management
    stop_loss: float = 0.02  # 2%
    take_profit: float = 0.05  # 5%
    max_drawdown_limit: float = 0.15  # 15%
    confidence_threshold: float = 0.7
    
    # Advanced features
    walk_forward_enabled: bool = False
    monte_carlo_enabled: bool = False
    monte_carlo_simulations: int = 1000
    
    # Config file path
    config_path: str = 'config/parameters.yaml'

@dataclass
class TradeResult:
    """Result of a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: int
    side: str  # 'long' or 'short'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    agent_signals: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Risk metrics
    value_at_risk: float
    conditional_var: float
    downside_deviation: float
    
    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    monthly_returns: pd.Series
    
    # Trade details
    trades: List[TradeResult]
    
    # Additional results
    walk_forward_results: Optional[Dict[str, Any]] = None
    monte_carlo_results: Optional[Dict[str, Any]] = None
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

class ComprehensiveBacktestEngine:
    """
    Unified backtesting engine combining:
    - Simple backtest logic from domain implementation
    - Enhanced features from enhanced_backtest_engine
    - Configuration-driven approach
    - Multi-symbol support
    - Comprehensive metrics
    """
    
    def __init__(self, config: Union[BacktestConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            self.config = BacktestConfig(**config)
        else:
            self.config = config
            
        # Load additional config from YAML if specified
        if self.config.config_path:
            try:
                with open(self.config.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    # Update config with YAML values
                    for key, value in yaml_config.get('backtest', {}).items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            except Exception as e:
                logger.warning(f"Could not load config from {self.config.config_path}: {e}")
        
        self.market_data_cache = {}
        self.benchmark_data = None
        
    def run_backtest(self, signal_generator: Callable, 
                    price_data: Optional[Dict[str, pd.DataFrame]] = None) -> BacktestResults:
        """
        Run comprehensive backtest
        
        Args:
            signal_generator: Function that generates trading signals
                            Should accept (symbol, historical_data, current_date) and return signal dict
            price_data: Optional pre-loaded price data. If None, will fetch from yfinance
        """
        logger.info(f"Starting backtest for {len(self.config.symbols)} symbols")
        
        # Get market data
        if price_data is None:
            market_data = self._fetch_all_market_data()
        else:
            market_data = price_data
            
        if not market_data:
            raise ValueError("No market data available for backtesting")
            
        # Run main backtest
        portfolio = Portfolio(self.config)
        all_trades = []
        
        # Get all trading dates
        all_dates = set()
        for data in market_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(all_dates)
        
        # Skip warmup period
        start_idx = 60  # Need history for indicators
        
        for date_idx in range(start_idx, len(trading_dates)):
            current_date = trading_dates[date_idx]
            
            # Check exits first
            closed_trades = portfolio.check_exits(current_date, market_data)
            all_trades.extend(closed_trades)
            
            # Generate signals for each symbol
            for symbol, data in market_data.items():
                if current_date not in data.index:
                    continue
                    
                # Skip if max positions reached
                if len(portfolio.open_positions) >= self.config.max_positions:
                    break
                    
                # Get historical data up to current date
                historical = data.loc[:current_date]
                
                # Generate signal
                try:
                    signal = signal_generator(symbol, historical, current_date)
                    
                    if signal and signal.get('confidence', 0) >= self.config.confidence_threshold:
                        # Execute trade
                        trade = portfolio.execute_trade(
                            signal, current_date, market_data[symbol]
                        )
                        
                except Exception as e:
                    logger.error(f"Signal generation error for {symbol} on {current_date}: {e}")
                    
            # Update portfolio value
            portfolio.update_value(current_date, market_data)
            
        # Close all positions at end
        final_trades = portfolio.close_all_positions(trading_dates[-1], market_data)
        all_trades.extend(final_trades)
        
        # Create results
        results = self._create_backtest_results(portfolio, all_trades)
        
        # Run Monte Carlo if enabled
        if self.config.monte_carlo_enabled and len(all_trades) > 10:
            results.monte_carlo_results = self._run_monte_carlo_simulations(all_trades)
            
        return results
        
    def _fetch_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols"""
        market_data = {}
        
        # Fetch benchmark (SPY)
        logger.info("Fetching benchmark data (SPY)")
        try:
            benchmark = yf.download('SPY', start=self.config.start_date, 
                                   end=self.config.end_date, progress=False)
            self.benchmark_data = benchmark
        except Exception as e:
            logger.warning(f"Could not fetch benchmark data: {e}")
        
        # Fetch symbol data in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for symbol in self.config.symbols:
                future = executor.submit(self._fetch_symbol_data, symbol)
                futures[symbol] = future
                
            for symbol, future in futures.items():
                try:
                    data = future.result()
                    if not data.empty:
                        market_data[symbol] = data
                        logger.info(f"Fetched {len(data)} days of data for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    
        return market_data
        
    def _fetch_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single symbol"""
        try:
            data = yf.download(symbol, start=self.config.start_date, 
                             end=self.config.end_date, progress=False)
            
            if data.empty:
                return pd.DataFrame()
                
            # Add basic indicators
            data['Returns'] = data['Close'].pct_change()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
            
    def _run_monte_carlo_simulations(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Run Monte Carlo simulations on trade results"""
        trade_returns = [t.pnl_percent for t in trades if t.exit_date]
        
        simulation_results = []
        for _ in range(self.config.monte_carlo_simulations):
            # Bootstrap sample trades
            simulated_trades = np.random.choice(
                trade_returns, 
                size=len(trade_returns), 
                replace=True
            )
            
            # Calculate metrics
            total_return = np.prod(1 + np.array(simulated_trades)) - 1
            simulation_results.append(total_return)
            
        return {
            'mean_return': np.mean(simulation_results),
            'std_return': np.std(simulation_results),
            'percentiles': {
                '5%': np.percentile(simulation_results, 5),
                '25%': np.percentile(simulation_results, 25),
                '50%': np.percentile(simulation_results, 50),
                '75%': np.percentile(simulation_results, 75),
                '95%': np.percentile(simulation_results, 95)
            }
        }
        
    def _create_backtest_results(self, portfolio: 'Portfolio', 
                               trades: List[TradeResult]) -> BacktestResults:
        """Create comprehensive backtest results"""
        equity_curve = pd.Series(portfolio.equity_history)
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Annualized metrics
        trading_days = len(equity_curve)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility and Sharpe
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown_series = (cumulative - running_max) / running_max
        max_drawdown = drawdown_series.min()
        
        # Trade statistics
        closed_trades = [t for t in trades if t.exit_date]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Monthly returns
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=0,  # Would need to calculate
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            value_at_risk=var_95,
            conditional_var=cvar_95,
            downside_deviation=downside_deviation,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            monthly_returns=monthly_returns,
            trades=trades
        )


class Portfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.equity_history = [config.initial_capital]
        self.open_positions = {}
        self.closed_trades = []
        
    def execute_trade(self, signal: Dict[str, Any], date: datetime,
                     price_data: pd.DataFrame) -> Optional[TradeResult]:
        """Execute a trade based on signal"""
        symbol = signal.get('symbol')
        if not symbol or symbol in self.open_positions:
            return None
            
        # Get current price
        if date not in price_data.index:
            return None
            
        current_bar = price_data.loc[date]
        entry_price = current_bar['Close']
        
        # Apply slippage
        side = signal.get('action', 'long')
        if side == 'long':
            entry_price *= (1 + self.config.slippage)
        else:
            entry_price *= (1 - self.config.slippage)
            
        # Calculate position size
        position_value = self.cash * self.config.position_size
        position_size = int(position_value / entry_price)
        
        if position_size <= 0:
            return None
            
        # Calculate commission
        commission = position_value * self.config.commission
        
        # Check if we have enough cash
        total_cost = position_value + commission
        if total_cost > self.cash:
            return None
            
        # Create trade
        trade = TradeResult(
            symbol=symbol,
            entry_date=date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_size,
            side=side,
            commission=commission,
            agent_signals=signal.get('agent_breakdown', {})
        )
        
        # Update portfolio
        self.cash -= total_cost
        self.open_positions[symbol] = trade
        
        return trade
        
    def check_exits(self, date: datetime, market_data: Dict[str, pd.DataFrame]) -> List[TradeResult]:
        """Check and execute exits for open positions"""
        closed_trades = []
        
        for symbol, trade in list(self.open_positions.items()):
            if symbol not in market_data or date not in market_data[symbol].index:
                continue
                
            current_bar = market_data[symbol].loc[date]
            current_price = current_bar['Close']
            
            # Calculate P&L
            if trade.side == 'long':
                pnl_percent = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_percent = (trade.entry_price - current_price) / trade.entry_price
                
            # Check exit conditions
            exit_reason = None
            
            # Stop loss
            if pnl_percent <= -self.config.stop_loss:
                exit_reason = "stop_loss"
                
            # Take profit
            elif pnl_percent >= self.config.take_profit:
                exit_reason = "take_profit"
                
            # Exit if we have a reason
            if exit_reason:
                trade = self._close_position(symbol, date, current_price, exit_reason)
                closed_trades.append(trade)
                
        return closed_trades
        
    def _close_position(self, symbol: str, date: datetime, 
                       exit_price: float, exit_reason: str) -> TradeResult:
        """Close a position"""
        trade = self.open_positions.pop(symbol)
        
        # Apply slippage
        if trade.side == 'long':
            exit_price *= (1 - self.config.slippage)
        else:
            exit_price *= (1 + self.config.slippage)
            
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        if trade.side == 'long':
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size
            
        trade.pnl_percent = trade.pnl / (trade.entry_price * trade.position_size)
        
        # Commission on exit
        exit_commission = exit_price * trade.position_size * self.config.commission
        trade.commission += exit_commission
        trade.pnl -= exit_commission
        
        # Update cash
        self.cash += (exit_price * trade.position_size - exit_commission)
        
        return trade
        
    def update_value(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Update portfolio value"""
        total_value = self.cash
        
        # Add value of open positions
        for symbol, trade in self.open_positions.items():
            if symbol in market_data and date in market_data[symbol].index:
                current_price = market_data[symbol].loc[date]['Close']
                position_value = current_price * trade.position_size
                total_value += position_value
                
        self.equity_history.append(total_value)
        
    def close_all_positions(self, date: datetime, 
                          market_data: Dict[str, pd.DataFrame]) -> List[TradeResult]:
        """Close all open positions"""
        closed_trades = []
        
        for symbol in list(self.open_positions.keys()):
            if symbol in market_data and date in market_data[symbol].index:
                current_price = market_data[symbol].loc[date]['Close']
                trade = self._close_position(symbol, date, current_price, "end_of_backtest")
                closed_trades.append(trade)
                
        return closed_trades 