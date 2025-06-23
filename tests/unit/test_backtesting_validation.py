"""
Tests for backtesting validation in GoldenSignalsAI V2.
Based on best practices for rigorous signal validation and backtesting.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestBacktestingValidation:
    """Test backtesting functionality and validation"""
    
    @pytest.fixture
    def historical_market_data(self):
        """Generate realistic historical market data for backtesting"""
        # Generate 1 year of daily data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        # Simulate realistic price movement with trends and volatility
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        
        # Add some trend components
        trend = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.001
        returns += trend
        
        # Create price series
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
            'low': prices * (1 + np.random.uniform(-0.01, 0, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def trading_signals(self):
        """Generate sample trading signals for backtesting"""
        signal_dates = pd.date_range('2023-01-15', '2023-12-15', freq='15D')
        
        signals = []
        for i, date in enumerate(signal_dates):
            signal = {
                'timestamp': date,
                'symbol': 'TEST',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'confidence': np.random.uniform(0.6, 0.95),
                'price_target': None,  # Will be set based on market data
                'stop_loss': None,
                'take_profit': None
            }
            signals.append(signal)
        
        return pd.DataFrame(signals)
    
    def test_backtest_engine_initialization(self):
        """Test backtesting engine setup and configuration"""
        class BacktestEngine:
            def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
                self.initial_capital = initial_capital
                self.commission = commission
                self.slippage = slippage
                self.positions = []
                self.trades = []
                self.equity_curve = []
                
            def validate_config(self):
                """Validate backtesting configuration"""
                assert self.initial_capital > 0, "Initial capital must be positive"
                assert 0 <= self.commission < 0.1, "Commission must be between 0 and 10%"
                assert 0 <= self.slippage < 0.01, "Slippage must be between 0 and 1%"
                return True
        
        # Test valid configuration
        engine = BacktestEngine()
        assert engine.validate_config()
        
        # Test invalid configurations
        with pytest.raises(AssertionError):
            invalid_engine = BacktestEngine(initial_capital=-1000)
            invalid_engine.validate_config()
    
    def test_realistic_trade_execution(self, historical_market_data):
        """Test realistic trade execution with costs and slippage"""
        data = historical_market_data
        
        def execute_trade(signal, market_data, commission=0.001, slippage=0.0005):
            """Simulate realistic trade execution"""
            # Find the market data for signal timestamp
            market_row = market_data[market_data['timestamp'] == signal['timestamp']].iloc[0]
            
            # Calculate execution price with slippage
            if signal['action'] == 'buy':
                # Buy at ask (slightly above market price)
                base_price = market_row['close']
                execution_price = base_price * (1 + slippage)
            else:
                # Sell at bid (slightly below market price)
                base_price = market_row['close']
                execution_price = base_price * (1 - slippage)
            
            # Calculate trade size (example: fixed position size)
            position_size = 10000 / execution_price  # $10,000 position
            
            # Calculate commission
            commission_cost = position_size * execution_price * commission
            
            # Calculate total cost
            if signal['action'] == 'buy':
                total_cost = (position_size * execution_price) + commission_cost
            else:
                total_cost = (position_size * execution_price) - commission_cost
            
            trade = {
                'timestamp': signal['timestamp'],
                'action': signal['action'],
                'size': position_size,
                'execution_price': execution_price,
                'commission': commission_cost,
                'total_cost': total_cost,
                'slippage_cost': abs(execution_price - base_price) * position_size
            }
            
            return trade
        
        # Test trade execution
        test_signal = {
            'timestamp': data['timestamp'].iloc[10],
            'action': 'buy'
        }
        
        trade = execute_trade(test_signal, data)
        
        # Verify trade execution
        assert trade['execution_price'] > 0
        assert trade['commission'] > 0
        assert trade['slippage_cost'] > 0
        assert trade['total_cost'] > trade['size'] * trade['execution_price']  # Includes costs
    
    def test_performance_metrics_calculation(self, historical_market_data, trading_signals):
        """Test calculation of key performance metrics"""
        data = historical_market_data
        signals = trading_signals
        
        # Simulate a simple backtest
        initial_capital = 100000
        equity = initial_capital
        equity_curve = [equity]
        trades = []
        position = None
        
        for _, signal in signals.iterrows():
            market_data = data[data['timestamp'] <= signal['timestamp']].iloc[-1]
            
            if signal['action'] == 'buy' and position is None:
                # Open position
                position = {
                    'entry_price': market_data['close'],
                    'entry_time': signal['timestamp'],
                    'size': 1000 / market_data['close']
                }
            elif signal['action'] == 'sell' and position is not None:
                # Close position
                exit_price = market_data['close']
                pnl = (exit_price - position['entry_price']) * position['size']
                equity += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': signal['timestamp'],
                    'pnl': pnl,
                    'return': pnl / (position['entry_price'] * position['size'])
                })
                
                position = None
                equity_curve.append(equity)
        
        # Calculate performance metrics
        def calculate_metrics(trades, equity_curve, initial_capital):
            """Calculate comprehensive performance metrics"""
            if not trades:
                return {}
            
            returns = [trade['return'] for trade in trades]
            equity_series = pd.Series(equity_curve)
            
            # Basic metrics
            total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
            win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
            
            # Sharpe ratio (annualized, assuming daily data)
            if len(equity_series) > 1:
                daily_returns = equity_series.pct_change().dropna()
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Profit factor
            gross_profits = sum(r for r in returns if r > 0)
            gross_losses = abs(sum(r for r in returns if r < 0))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            metrics = {
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'num_trades': len(trades)
            }
            
            return metrics
        
        # Calculate and verify metrics
        metrics = calculate_metrics(trades, equity_curve, initial_capital)
        
        # Verify metric calculations
        assert 'total_return' in metrics
        assert 'win_rate' in metrics
        assert 0 <= metrics['win_rate'] <= 1
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['max_drawdown'] <= 0  # Drawdown is negative
        assert 'profit_factor' in metrics
        assert metrics['profit_factor'] >= 0
    
    def test_walk_forward_optimization(self, historical_market_data):
        """Test walk-forward optimization for realistic backtesting"""
        data = historical_market_data
        
        def walk_forward_backtest(data, window_size=90, step_size=30):
            """Implement walk-forward optimization"""
            results = []
            
            for i in range(window_size, len(data) - step_size, step_size):
                # Training window
                train_start = i - window_size
                train_end = i
                train_data = data.iloc[train_start:train_end]
                
                # Test window
                test_start = i
                test_end = min(i + step_size, len(data))
                test_data = data.iloc[test_start:test_end]
                
                # Simulate model training and prediction
                # (In reality, you would train your model here)
                mock_model_performance = np.random.uniform(0.4, 0.7)
                
                result = {
                    'train_period': (train_data['timestamp'].iloc[0], train_data['timestamp'].iloc[-1]),
                    'test_period': (test_data['timestamp'].iloc[0], test_data['timestamp'].iloc[-1]),
                    'performance': mock_model_performance,
                    'num_train_samples': len(train_data),
                    'num_test_samples': len(test_data)
                }
                
                results.append(result)
            
            return results
        
        # Run walk-forward optimization
        wf_results = walk_forward_backtest(data)
        
        # Verify walk-forward results
        assert len(wf_results) > 0
        for result in wf_results:
            assert result['num_train_samples'] > 0
            assert result['num_test_samples'] > 0
            assert 0 <= result['performance'] <= 1
            
            # Verify no overlap between train and test periods
            train_end = result['train_period'][1]
            test_start = result['test_period'][0]
            assert train_end < test_start
    
    def test_stress_testing_scenarios(self, historical_market_data):
        """Test backtesting under extreme market conditions"""
        data = historical_market_data.copy()
        
        # Create stress scenarios
        def apply_stress_scenario(data, scenario_type):
            """Apply stress scenarios to market data"""
            stressed_data = data.copy()
            
            if scenario_type == 'flash_crash':
                # Simulate a 10% flash crash
                crash_idx = len(data) // 2
                stressed_data.loc[crash_idx:crash_idx+5, 'close'] *= 0.9
                stressed_data.loc[crash_idx:crash_idx+5, 'low'] *= 0.85
                stressed_data.loc[crash_idx:crash_idx+5, 'volume'] *= 5
                
            elif scenario_type == 'high_volatility':
                # Triple the volatility
                returns = stressed_data['close'].pct_change()
                mean_return = returns.mean()
                amplified_returns = mean_return + (returns - mean_return) * 3
                stressed_data['close'] = stressed_data['close'].iloc[0] * (1 + amplified_returns).cumprod()
                
            elif scenario_type == 'low_liquidity':
                # Reduce volume by 90%
                stressed_data['volume'] *= 0.1
                # Increase bid-ask spread (simulated through price movements)
                stressed_data['high'] *= 1.005
                stressed_data['low'] *= 0.995
            
            return stressed_data
        
        # Test different stress scenarios
        scenarios = ['flash_crash', 'high_volatility', 'low_liquidity']
        
        for scenario in scenarios:
            stressed_data = apply_stress_scenario(data, scenario)
            
            # Verify stress was applied
            if scenario == 'flash_crash':
                # Check the specific crash period rather than overall minimum
                crash_idx = len(data) // 2
                crash_period_prices = stressed_data.loc[crash_idx:crash_idx+5, 'close']
                normal_period_prices = data.loc[crash_idx:crash_idx+5, 'close']
                assert crash_period_prices.mean() < normal_period_prices.mean() * 0.92
                
            elif scenario == 'high_volatility':
                stressed_vol = stressed_data['close'].pct_change().std()
                normal_vol = data['close'].pct_change().std()
                assert stressed_vol > normal_vol * 2
                
            elif scenario == 'low_liquidity':
                assert stressed_data['volume'].mean() < data['volume'].mean() * 0.2
    
    def test_data_snooping_prevention(self):
        """Test measures to prevent data snooping and look-ahead bias"""
        # Create time-series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.random.randn(100).cumsum()
        
        data = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
        
        def calculate_signal_with_lookahead(data, idx):
            """Example of what NOT to do - uses future data"""
            if idx >= len(data) - 1:
                return None
            
            # BAD: Looking at tomorrow's price
            future_price = data.iloc[idx + 1]['price']
            current_price = data.iloc[idx]['price']
            
            return 'buy' if future_price > current_price else 'sell'
        
        def calculate_signal_properly(data, idx, lookback=10):
            """Proper signal calculation using only past data"""
            if idx < lookback:
                return None
            
            # GOOD: Only using historical data
            historical_prices = data.iloc[idx-lookback:idx]['price']
            current_price = data.iloc[idx]['price']
            
            # Simple momentum strategy
            avg_price = historical_prices.mean()
            return 'buy' if current_price > avg_price else 'sell'
        
        # Test for look-ahead bias detection
        lookahead_signals = []
        proper_signals = []
        
        for i in range(len(data)):
            lookahead_signals.append(calculate_signal_with_lookahead(data, i))
            proper_signals.append(calculate_signal_properly(data, i))
        
        # In a real test, you would verify that lookahead signals perform "too well"
        # and flag this as a potential bias
        assert len(proper_signals) == len(data)
        assert proper_signals[:10] == [None] * 10  # No signals for first 10 days
    
    def test_out_of_sample_validation(self, historical_market_data, trading_signals):
        """Test proper out-of-sample validation"""
        data = historical_market_data
        
        # Split data properly
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size+val_size]
        test_data = data.iloc[train_size+val_size:]
        
        # Ensure no overlap
        assert len(set(train_data.index) & set(val_data.index)) == 0
        assert len(set(train_data.index) & set(test_data.index)) == 0
        assert len(set(val_data.index) & set(test_data.index)) == 0
        
        # Verify chronological order
        assert train_data['timestamp'].max() < val_data['timestamp'].min()
        assert val_data['timestamp'].max() < test_data['timestamp'].min()
        
        # Mock performance on each set
        def evaluate_strategy(data):
            """Mock strategy evaluation"""
            # In reality, this would run your actual strategy
            return {
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'win_rate': np.random.uniform(0.4, 0.6),
                'max_drawdown': np.random.uniform(-0.3, -0.05)
            }
        
        train_perf = evaluate_strategy(train_data)
        val_perf = evaluate_strategy(val_data)
        test_perf = evaluate_strategy(test_data)
        
        # Log performance degradation (common in real scenarios)
        print(f"Train Sharpe: {train_perf['sharpe_ratio']:.2f}")
        print(f"Val Sharpe: {val_perf['sharpe_ratio']:.2f}")
        print(f"Test Sharpe: {test_perf['sharpe_ratio']:.2f}")
        
        # All sets should have valid performance metrics
        for perf in [train_perf, val_perf, test_perf]:
            assert perf['sharpe_ratio'] > 0
            assert 0 < perf['win_rate'] < 1
            assert perf['max_drawdown'] < 0 