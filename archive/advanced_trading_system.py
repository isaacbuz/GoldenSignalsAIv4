"""
Advanced example demonstrating the complete trading system with multiple agents and backtesting.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from agents.orchestration.orchestrator import AgentOrchestrator
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.backtesting.backtest_engine import BacktestEngine

def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical market data"""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    return data

def fetch_news_data(symbol: str, start_date: str, end_date: str) -> List[str]:
    """Simulate news data (replace with actual news API in production)"""
    # For demonstration, generate synthetic news
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sentiments = np.random.normal(0, 1, size=len(dates))
    
    texts = []
    for date, sentiment in zip(dates, sentiments):
        if sentiment > 0.5:
            text = f"Positive outlook for {symbol} as market conditions improve"
        elif sentiment < -0.5:
            text = f"Concerns grow over {symbol}'s market position"
        else:
            text = f"Mixed market signals for {symbol}"
        texts.append(text)
    
    return texts

def plot_backtest_results(prices: pd.Series, results: Dict[str, Any]) -> None:
    """Plot backtest results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot price and trades
    ax1.plot(prices.index, prices.values, label='Price', alpha=0.7)
    
    # Plot buy/sell points
    for trade in results['trades']:
        if trade['action'] == 'buy':
            ax1.scatter(trade['timestamp'], trade['price'], 
                       color='green', marker='^', s=100)
        elif trade['action'] == 'sell':
            ax1.scatter(trade['timestamp'], trade['price'], 
                       color='red', marker='v', s=100)
    
    ax1.set_title('Price and Trades')
    ax1.legend()
    
    # Plot equity curve
    equity_data = pd.DataFrame(results['equity_curve'])
    ax2.plot(equity_data['timestamp'], equity_data['equity'], label='Portfolio Value')
    ax2.set_title('Equity Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def print_backtest_statistics(results: Dict[str, Any]) -> None:
    """Print backtest statistics"""
    print("\nBacktest Statistics:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Trades: {len(results['trades'])}")

def main():
    # Initialize orchestrator and agents
    orchestrator = AgentOrchestrator()
    
    # Create and register agents
    rsi_agent = RSIAgent(
        name="RSI_14",
        period=14,
        overbought=70,
        oversold=30
    )
    
    macd_agent = MACDAgent(
        name="MACD_12_26_9",
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    sentiment_agent = SentimentAgent(name="Sentiment_NLTK")
    
    # Register all agents
    for agent in [rsi_agent, macd_agent, sentiment_agent]:
        orchestrator.register_agent(agent)
    
    # Configure signal weights
    orchestrator.update_signal_weights({
        "technical": 0.6,  # Higher weight for technical signals
        "sentiment": 0.4   # Lower weight for sentiment
    })
    
    # Fetch market data
    symbol = "AAPL"  # Example using Apple stock
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    market_data = fetch_market_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                  end_date.strftime('%Y-%m-%d'))
    news_data = fetch_news_data(symbol, start_date.strftime('%Y-%m-%d'), 
                              end_date.strftime('%Y-%m-%d'))
    
    # Initialize and run backtest
    backtest = BacktestEngine(
        orchestrator=orchestrator,
        initial_capital=100000.0,
        commission=0.001
    )
    
    results = backtest.run(
        prices=market_data['Close'],
        texts=news_data,
        window=100
    )
    
    # Plot and print results
    plot_backtest_results(market_data['Close'], results)
    print_backtest_statistics(results)

if __name__ == "__main__":
    main() 