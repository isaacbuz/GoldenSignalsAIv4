"""
Tests for the backtesting engine.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from agents.backtesting.backtest_engine import BacktestEngine

def test_backtest_initialization(orchestrator):
    """Test backtest engine initialization"""
    engine = BacktestEngine(
        orchestrator=orchestrator,
        initial_capital=100000.0,
        commission=0.001
    )
    assert engine.initial_capital == 100000.0
    assert engine.commission == 0.001
    assert engine.capital == 100000.0
    assert engine.position == 0
    assert len(engine.trades) == 0
    assert len(engine.equity_curve) == 0

def test_market_data_preparation(backtest_engine, historical_data):
    """Test market data preparation"""
    window = 100
    data = backtest_engine.prepare_market_data(
        prices=historical_data["Close"],
        texts=["Test news"] * len(historical_data),
        window=window
    )
    
    assert "close_prices" in data
    assert "texts" in data
    assert "timestamp" in data
    assert len(data["close_prices"]) == window
    assert isinstance(data["timestamp"], str)

def test_trade_execution(backtest_engine):
    """Test trade execution logic"""
    # Test buy execution
    backtest_engine.execute_trade(
        price=100.0,
        action="buy",
        confidence=0.8,
        timestamp=datetime.now()
    )
    assert backtest_engine.position > 0
    assert len(backtest_engine.trades) == 1
    assert backtest_engine.trades[0]["action"] == "buy"
    
    # Test sell execution
    backtest_engine.execute_trade(
        price=110.0,
        action="sell",
        confidence=0.7,
        timestamp=datetime.now()
    )
    assert backtest_engine.position == 0
    assert len(backtest_engine.trades) == 2
    assert backtest_engine.trades[1]["action"] == "sell"
    
    # Verify profit calculation
    assert backtest_engine.trades[1]["profit"] > 0

def test_equity_calculation(backtest_engine):
    """Test equity calculation"""
    # Initial equity
    equity = backtest_engine.calculate_equity(100.0)
    assert equity == backtest_engine.initial_capital
    
    # Buy position
    backtest_engine.execute_trade(
        price=100.0,
        action="buy",
        confidence=1.0,
        timestamp=datetime.now()
    )
    
    # Calculate equity with price increase
    equity = backtest_engine.calculate_equity(110.0)
    assert equity > backtest_engine.initial_capital

def test_backtest_run(backtest_engine, historical_data):
    """Test full backtest run"""
    results = backtest_engine.run(
        prices=historical_data["Close"],
        texts=["Market update"] * len(historical_data),
        window=100
    )
    
    assert "total_return" in results
    assert "annual_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "win_rate" in results
    assert "profit_factor" in results
    assert "trades" in results
    assert "equity_curve" in results

def test_statistics_calculation(backtest_engine, historical_data):
    """Test statistics calculation"""
    # Run backtest
    backtest_engine.run(
        prices=historical_data["Close"],
        texts=["Market update"] * len(historical_data),
        window=100
    )
    
    # Generate some trades
    for i in range(10):
        price = 100.0 * (1 + i/100)
        backtest_engine.execute_trade(
            price=price,
            action="buy" if i % 2 == 0 else "sell",
            confidence=0.8,
            timestamp=datetime.now()
        )
    
    # Calculate statistics
    results = backtest_engine.calculate_statistics(pd.DataFrame())
    
    assert -1 <= results["total_return"] <= 1
    assert -1 <= results["annual_return"] <= 1
    assert isinstance(results["sharpe_ratio"], float)
    assert -1 <= results["max_drawdown"] <= 0
    assert 0 <= results["win_rate"] <= 1
    assert results["profit_factor"] >= 0

def test_commission_impact(orchestrator):
    """Test impact of different commission rates"""
    # Create engines with different commission rates
    low_commission = BacktestEngine(orchestrator, commission=0.001)
    high_commission = BacktestEngine(orchestrator, commission=0.01)
    
    # Generate test data
    prices = pd.Series([100.0 * (1 + i/100) for i in range(100)])
    texts = ["Update"] * len(prices)
    
    # Run backtests
    low_results = low_commission.run(prices, texts)
    high_results = high_commission.run(prices, texts)
    
    # Higher commission should result in lower returns
    assert low_results["total_return"] > high_results["total_return"]

def test_edge_cases(backtest_engine):
    """Test edge cases and error handling"""
    # Test with no trades
    results = backtest_engine.calculate_statistics(pd.DataFrame())
    assert results["total_return"] == 0.0
    assert results["win_rate"] == 0.0
    
    # Test with single price
    with pytest.raises(ValueError):
        backtest_engine.run(pd.Series([100.0]), ["Test"])
    
    # Test with mismatched data lengths
    with pytest.raises(ValueError):
        backtest_engine.run(
            pd.Series([100.0] * 100),
            ["Test"] * 50  # Different length
        ) 