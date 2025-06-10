import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.domain.trading.data_manager import DataManager
from src.domain.models.model_manager import ModelManager
from src.domain.portfolio.portfolio_manager import PortfolioManager
from src.domain.analytics.analytics_manager import AnalyticsManager

class TestIntegration:
    @pytest.fixture
    def setup_managers(self):
        """Set up all managers needed for integration testing"""
        data_manager = DataManager()
        model_manager = ModelManager(model_dir="test_models")
        portfolio_manager = PortfolioManager(initial_capital=100000.0)
        analytics_manager = AnalyticsManager()
        return data_manager, model_manager, portfolio_manager, analytics_manager
    
    @pytest.fixture
    def sample_market_data(self, setup_managers):
        """Get sample market data for testing"""
        data_manager = setup_managers[0]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        symbols = ["AAPL", "MSFT"]
        return data_manager.fetch_market_data(symbols, start_date, end_date)

    def test_end_to_end_workflow(self, setup_managers):
        """Test complete workflow from data to analytics"""
        data_manager, model_manager, portfolio_manager, analytics_manager = setup_managers
        
        # 1. Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        symbol = "AAPL"
        data = data_manager.fetch_market_data(symbol, start_date, end_date)
        
        # 2. Prepare features and target
        df = data[symbol]
        feature_columns = ['Open', 'High', 'Low', 'Volume']
        target_column = 'Close'
        features, target = data_manager.prepare_features(
            df, feature_columns, target_column, lookback_periods=5
        )
        
        # 3. Split data and train model
        X_train, X_test, y_train, y_test = data_manager.split_data(features, target)
        model = model_manager.train_model(
            'test_model',
            X_train,
            y_train,
            model_type='regressor'
        )
        
        # 4. Make predictions and evaluate
        predictions = model_manager.predict('test_model', X_test)
        metrics = model_manager.evaluate_model('test_model', X_test, y_test)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        
        # 5. Place trades based on predictions
        last_price = df['Close'].iloc[-1]
        if predictions[-1] > last_price:
            success = portfolio_manager.place_order(symbol, 100, last_price)
            assert success
        
        # 6. Calculate portfolio metrics
        current_prices = {symbol: last_price}
        portfolio_metrics = portfolio_manager.get_portfolio_metrics(current_prices)
        assert 'total_value' in portfolio_metrics
        assert 'unrealized_pnl' in portfolio_metrics
        
        # 7. Calculate analytics
        returns = analytics_manager.calculate_returns(df['Close'])
        risk_metrics = analytics_manager.calculate_risk_metrics(returns)
        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        
        # 8. Calculate drawdowns
        drawdowns = analytics_manager.calculate_drawdowns(returns)
        assert len(drawdowns) > 0
        assert 'drawdown' in drawdowns.columns

    def test_multi_asset_portfolio(self, setup_managers, sample_market_data):
        """Test portfolio management with multiple assets"""
        _, _, portfolio_manager, analytics_manager = setup_managers
        
        # Place orders for multiple assets
        symbols = ["AAPL", "MSFT"]
        orders = [
            ("AAPL", 100, sample_market_data["AAPL"]['Close'].iloc[-1]),
            ("MSFT", 50, sample_market_data["MSFT"]['Close'].iloc[-1])
        ]
        
        for symbol, quantity, price in orders:
            success = portfolio_manager.place_order(symbol, quantity, price)
            assert success
        
        # Verify portfolio positions
        positions = portfolio_manager.get_position_sizes()
        assert len(positions) == 2
        assert positions["AAPL"] == 100
        assert positions["MSFT"] == 50
        
        # Calculate portfolio value
        current_prices = {
            symbol: data['Close'].iloc[-1]
            for symbol, data in sample_market_data.items()
        }
        metrics = portfolio_manager.get_portfolio_metrics(current_prices)
        assert metrics['total_value'] > 0

    def test_model_persistence(self, setup_managers, sample_market_data):
        """Test model saving and loading"""
        data_manager, model_manager, _, _ = setup_managers
        
        # Prepare data and train model
        df = sample_market_data["AAPL"]
        features, target = data_manager.prepare_features(
            df, ['Open', 'High', 'Low', 'Volume'], 'Close'
        )
        X_train, X_test, y_train, y_test = data_manager.split_data(features, target)
        
        # Train and save model
        model_id = 'persistence_test_model'
        model_manager.train_model(model_id, X_train, y_train, model_type='regressor')
        
        # Create new model manager and load model
        new_model_manager = ModelManager(model_dir="test_models")
        predictions = new_model_manager.predict(model_id, X_test)
        assert len(predictions) == len(y_test)

    def test_risk_analytics(self, setup_managers, sample_market_data):
        """Test risk analytics calculations"""
        _, _, _, analytics_manager = setup_managers
        
        # Calculate returns for multiple assets
        returns_data = {}
        for symbol, data in sample_market_data.items():
            returns = analytics_manager.calculate_returns(data['Close'])
            returns_data[symbol] = returns
        
        # Calculate and compare risk metrics
        metrics = {}
        for symbol, returns in returns_data.items():
            metrics[symbol] = analytics_manager.calculate_risk_metrics(returns)
        
        # Verify metrics
        for symbol_metrics in metrics.values():
            assert 'volatility' in symbol_metrics
            assert 'sharpe_ratio' in symbol_metrics
            assert 'max_drawdown' in symbol_metrics
            assert symbol_metrics['volatility'] > 0
        
        # Calculate portfolio metrics using AAPL as benchmark
        portfolio_metrics = analytics_manager.calculate_portfolio_metrics(
            returns_data["MSFT"],
            returns_data["AAPL"]
        )
        assert 'alpha' in portfolio_metrics
        assert 'beta' in portfolio_metrics
        assert 'information_ratio' in portfolio_metrics 