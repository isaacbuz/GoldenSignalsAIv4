# MCP Week 4: Portfolio Management Server

## Overview

Week 4 implements a comprehensive Portfolio Management Server that provides:
- Real-time portfolio tracking and position management
- Risk-based position sizing using Kelly Criterion
- Portfolio risk metrics and exposure analysis
- Trade execution with risk management
- Performance analytics including Sharpe ratio and drawdown
- Portfolio rebalancing recommendations

## Features

### 1. Portfolio Status Tracking
- Real-time portfolio valuation
- Position-level P&L tracking
- Cash management
- Portfolio allocation percentages

### 2. Position Sizing
- Kelly Criterion-based position sizing
- Volatility adjustments
- Risk limit enforcement
- Stop loss and take profit calculations

### 3. Trade Execution
- Buy/sell order execution
- Position averaging
- Risk validation
- Trade history tracking

### 4. Risk Management
- Portfolio-level risk metrics
- Position concentration analysis
- Value at Risk (VaR) calculations
- Risk limit monitoring

### 5. Performance Analytics
- Return calculations
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor

### 6. Portfolio Rebalancing
- Target allocation analysis
- Rebalancing recommendations
- Trade impact estimation

## Installation

The portfolio server is already configured in your `claude_desktop_config.json`:

```json
{
    "goldensignals-portfolio": {
        "command": "/path/to/.venv/bin/python",
        "args": ["/path/to/mcp_servers/portfolio_management_server.py"]
    }
}
```

## Usage Examples

### Get Portfolio Status

```python
# Tool: get_portfolio_status
# Returns current portfolio value, positions, and allocation

{
    "timestamp": "2025-06-17T17:15:34.630433",
    "cash": 83678.50,
    "total_market_value": 16321.50,
    "total_value": 100000.00,
    "total_return": 0.0,
    "positions": [
        {
            "symbol": "AAPL",
            "quantity": 93,
            "entry_price": 175.50,
            "current_price": 175.50,
            "market_value": 16321.50,
            "unrealized_pnl": 0.0,
            "pnl_percentage": 0.0
        }
    ],
    "allocation": {
        "CASH": 83.68,
        "AAPL": 16.32
    }
}
```

### Calculate Position Size

```python
# Tool: calculate_position_size
# Arguments: symbol, price, signal_confidence, volatility

{
    "symbol": "AAPL",
    "recommended_shares": 93,
    "position_value": 16321.50,
    "position_percentage": 16.32,
    "stop_loss": 169.18,
    "take_profit": 188.14,
    "risk_amount": 587.57,
    "risk_percentage": 0.59,
    "kelly_fraction": 0.775,
    "adjusted_fraction": 0.164
}
```

### Execute Trade

```python
# Tool: execute_trade
# Arguments: symbol, action, quantity, price, stop_loss, take_profit

{
    "status": "executed",
    "trade": {
        "timestamp": "2025-06-17T17:15:34",
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 93,
        "price": 175.50,
        "value": 16321.50
    },
    "portfolio_cash": 83678.50,
    "portfolio_value": 100000.00
}
```

### Get Risk Metrics

```python
# Tool: get_risk_metrics
# Returns portfolio risk analysis

{
    "total_risk_amount": 986.76,
    "total_risk_percentage": 0.99,
    "position_risks": [
        {
            "symbol": "AAPL",
            "risk_amount": 587.76,
            "risk_percentage": 0.59,
            "position_weight": 16.32
        }
    ],
    "concentration_risk": 0.505,
    "var_95": 3290.00,
    "var_95_percentage": 3.29
}
```

### Rebalance Portfolio

```python
# Tool: rebalance_portfolio
# Arguments: target_allocations (dict of symbol: percentage)

{
    "current_allocation": {
        "CASH": 70.53,
        "AAPL": 16.11,
        "MSFT": 13.36
    },
    "target_allocation": {
        "AAPL": 30.0,
        "MSFT": 30.0,
        "GOOGL": 20.0,
        "CASH": 20.0
    },
    "recommendations": [
        {
            "symbol": "AAPL",
            "action": "buy",
            "shares": 80,
            "value_change": 13863.42
        }
    ]
}
```

### Get Performance Analytics

```python
# Tool: get_performance_analytics
# Arguments: period_days (optional, default 30)

{
    "period_days": 30,
    "performance": {
        "total_return": 5.25,
        "annualized_return": 63.0,
        "volatility": 15.2,
        "sharpe_ratio": 2.85,
        "max_drawdown": 3.5
    },
    "trading_stats": {
        "total_trades": 45,
        "win_rate": 62.2,
        "average_win": 250.50,
        "average_loss": -125.30,
        "profit_factor": 2.15
    }
}
```

## Resources

The server provides three resources for monitoring:

1. **portfolio://status/live** - Real-time portfolio status
2. **portfolio://risk/metrics** - Current risk metrics
3. **portfolio://performance/history** - Historical performance data

## Risk Parameters

Default risk parameters:
- Maximum position size: 20% of portfolio
- Maximum risk per trade: 2% of portfolio
- Maximum portfolio risk: 6% total
- Volatility window: 20 periods
- Correlation threshold: 0.7

## Position Sizing Algorithm

The server uses a modified Kelly Criterion:

1. **Kelly Fraction**: `f = (p*b - q) / b`
   - p = probability of win (signal confidence)
   - b = win/loss ratio
   - q = 1 - p

2. **Safety Factor**: Uses 25% of Kelly fraction for conservative sizing

3. **Volatility Adjustment**: `vol_factor = 1 / (1 + volatility * 10)`

4. **Risk Limits**: Ensures position doesn't exceed max risk per trade

## Testing

Run the test script to verify functionality:

```bash
python test_portfolio_server.py
```

The test covers:
- Portfolio status retrieval
- Position sizing calculations
- Trade execution
- Risk metrics
- Rebalancing recommendations
- Performance analytics

## Integration with Trading System

The portfolio server integrates with:
- **Week 1**: Receives trading signals for position sizing
- **Week 2**: Uses market data for current prices
- **Week 3**: Processes agent signals for trade decisions

## Best Practices

1. **Position Sizing**: Always use `calculate_position_size` before executing trades
2. **Risk Monitoring**: Check `get_risk_metrics` regularly
3. **Rebalancing**: Review rebalancing recommendations weekly/monthly
4. **Performance Review**: Monitor performance analytics for strategy validation

## Troubleshooting

### Common Issues

1. **Insufficient Cash Error**
   - Check portfolio cash before executing buys
   - Consider partial positions or selling other holdings

2. **Position Size Limit**
   - Reduce signal confidence or adjust risk parameters
   - Consider portfolio diversification

3. **Risk Limit Exceeded**
   - Review current portfolio risk metrics
   - Adjust stop losses or reduce positions

## Future Enhancements

1. **Tax Optimization**
   - Tax-loss harvesting
   - Wash sale rule compliance
   - Long/short term gain optimization

2. **Advanced Risk Models**
   - Correlation matrix from actual data
   - Monte Carlo simulations
   - Stress testing scenarios

3. **Multi-Currency Support**
   - FX risk management
   - Cross-currency positions

4. **Options Integration**
   - Options position tracking
   - Greeks calculation
   - Complex strategies support

## Configuration

To modify risk parameters, edit the server initialization:

```python
self.risk_params = {
    "max_position_size": 0.20,      # Adjust max position size
    "max_risk_per_trade": 0.02,     # Adjust per-trade risk
    "max_portfolio_risk": 0.06,     # Adjust total portfolio risk
    "volatility_window": 20,        # Adjust volatility calculation window
    "correlation_threshold": 0.7    # Adjust correlation threshold
}
```

## Example Workflow

1. **Signal Reception**: Receive buy signal from agents
2. **Position Sizing**: Calculate optimal position size
3. **Risk Check**: Verify risk metrics are within limits
4. **Trade Execution**: Execute trade with stop loss/take profit
5. **Monitoring**: Track position performance
6. **Rebalancing**: Adjust portfolio based on targets
7. **Performance Review**: Analyze results and refine strategy 