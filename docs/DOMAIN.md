# Domain Components Documentation

## Overview
The domain layer contains the core business logic for the trading system. It is organized into four main components:
- Data Management
- Model Management
- Portfolio Management
- Analytics

## Data Management
The `DataManager` class in `domain/trading/data_manager.py` handles all market data operations.

### Key Features
- Market data fetching with caching
- Feature preparation for machine learning
- Data splitting for model training

### Example Usage
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

## Model Management
The `ModelManager` class in `domain/models/model_manager.py` handles machine learning model operations.

### Key Features
- Model training and persistence
- Prediction generation
- Model evaluation
- Automatic model saving and loading

### Example Usage
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

## Portfolio Management
The `PortfolioManager` class in `domain/portfolio/portfolio_manager.py` handles trading and portfolio operations.

### Key Features
- Order placement and position tracking
- Portfolio value calculation
- Position size management
- Trade history tracking

### Example Usage
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

## Analytics
The `AnalyticsManager` class in `domain/analytics/analytics_manager.py` provides financial analytics and risk metrics.

### Key Features
- Returns calculation (arithmetic and logarithmic)
- Risk metrics (volatility, Sharpe ratio, etc.)
- Portfolio performance metrics (alpha, beta, etc.)
- Drawdown analysis

### Example Usage
```python
analytics_manager = AnalyticsManager()

# Calculate returns
returns = analytics_manager.calculate_returns(prices, method='arithmetic')

# Get risk metrics
risk_metrics = analytics_manager.calculate_risk_metrics(returns)

# Analyze drawdowns
drawdowns = analytics_manager.calculate_drawdowns(returns, top_n=5)
```

## Integration Example
The components are designed to work together seamlessly:

```python
# 1. Fetch and prepare data
data = data_manager.fetch_market_data("AAPL", start_date, end_date)
features, target = data_manager.prepare_features(data["AAPL"], feature_cols, "Close")

# 2. Train model and make predictions
model = model_manager.train_model("aapl_model", X_train, y_train)
predictions = model_manager.predict("aapl_model", X_test)

# 3. Execute trades based on predictions
if predictions[-1] > current_price:
    portfolio_manager.place_order("AAPL", 100, current_price)

# 4. Analyze performance
returns = analytics_manager.calculate_returns(prices)
metrics = analytics_manager.calculate_risk_metrics(returns)
```

## Testing
Comprehensive integration tests are available in `tests/test_integration.py`. These tests cover:
- End-to-end workflow
- Multi-asset portfolio management
- Model persistence
- Risk analytics

To run the tests:
```bash
pytest tests/test_integration.py -v
``` 