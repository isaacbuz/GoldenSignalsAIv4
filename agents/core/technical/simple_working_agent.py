"""
Simple Working RSI Agent - MVP Example

This is a complete, working example of a trading agent that:
1. Fetches real market data using yfinance
2. Calculates RSI (Relative Strength Index)
3. Generates BUY/SELL/HOLD signals
4. Works with the existing framework

Use this as a template for other agents!
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Set up logging
logger = logging.getLogger(__name__)

class SimpleRSIAgent:
    """
    A simple but complete RSI-based trading agent.

    Trading Logic:
    - BUY when RSI < 30 (oversold)
    - SELL when RSI > 70 (overbought)
    - HOLD otherwise
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.name = "simple_rsi_agent"
        self.period = period
        self.oversold_threshold = oversold
        self.overbought_threshold = overbought
        logger.info(f"Initialized {self.name} with RSI period={period}")

    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.period

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def fetch_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Fetch historical data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                logger.warning(f"No data fetched for {symbol}")
                return None

            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate trading signal for a symbol

        Returns:
            Dict containing:
            - action: 'BUY', 'SELL', or 'HOLD'
            - confidence: 0.0 to 1.0
            - metadata: Additional information
        """
        try:
            # Fetch market data
            data = self.fetch_data(symbol)
            if data is None or len(data) < self.period + 1:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Insufficient data",
                        "agent": self.name
                    }
                }

            # Calculate RSI
            rsi_series = self.calculate_rsi(data['Close'])
            current_rsi = rsi_series.iloc[-1]

            # Skip if RSI is NaN
            if pd.isna(current_rsi):
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "RSI calculation failed",
                        "agent": self.name
                    }
                }

            # Generate signal based on RSI
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price * 100

            # Determine action and confidence
            if current_rsi < self.oversold_threshold:
                action = "BUY"
                # Confidence increases as RSI gets lower
                confidence = min(0.95, (self.oversold_threshold - current_rsi) / 30 + 0.6)
                reasoning = f"RSI at {current_rsi:.2f} indicates oversold condition"

            elif current_rsi > self.overbought_threshold:
                action = "SELL"
                # Confidence increases as RSI gets higher
                confidence = min(0.95, (current_rsi - self.overbought_threshold) / 30 + 0.6)
                reasoning = f"RSI at {current_rsi:.2f} indicates overbought condition"

            else:
                action = "HOLD"
                # Lower confidence in the neutral zone
                confidence = 0.3
                reasoning = f"RSI at {current_rsi:.2f} is in neutral territory"

            # Calculate additional indicators for context
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = data['Volume'].iloc[-1] / volume_avg if volume_avg > 0 else 1

            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "current_price": float(current_price),
                    "price_change_pct": float(price_change),
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "rsi": float(current_rsi),
                        "rsi_threshold_oversold": self.oversold_threshold,
                        "rsi_threshold_overbought": self.overbought_threshold,
                        "sma_20": float(sma_20) if not pd.isna(sma_20) else None,
                        "volume_ratio": float(volume_ratio),
                        "price": float(current_price)
                    },
                    "chart_data": {
                        "rsi_history": rsi_series.tail(50).fillna(50).tolist(),
                        "price_history": data['Close'].tail(50).tolist()
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {
                    "error": str(e),
                    "agent": self.name
                }
            }

    def backtest(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Simple backtest of the strategy
        """
        try:
            data = self.fetch_data(symbol, period)
            if data is None or len(data) < self.period + 1:
                return {"error": "Insufficient data for backtest"}

            # Calculate RSI for entire period
            rsi = self.calculate_rsi(data['Close'])

            # Generate signals
            signals = []
            for i in range(self.period, len(data)):
                if pd.notna(rsi.iloc[i]):
                    if rsi.iloc[i] < self.oversold_threshold:
                        signals.append(1)  # Buy
                    elif rsi.iloc[i] > self.overbought_threshold:
                        signals.append(-1)  # Sell
                    else:
                        signals.append(0)  # Hold
                else:
                    signals.append(0)

            # Calculate returns
            data_subset = data.iloc[self.period:].copy()
            data_subset['Signal'] = signals
            data_subset['Returns'] = data_subset['Close'].pct_change()
            data_subset['Strategy_Returns'] = data_subset['Signal'].shift(1) * data_subset['Returns']

            # Performance metrics
            total_return = (data_subset['Strategy_Returns'] + 1).prod() - 1
            buy_signals = sum(1 for s in signals if s == 1)
            sell_signals = sum(1 for s in signals if s == -1)

            return {
                "symbol": symbol,
                "period": period,
                "total_return": float(total_return * 100),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "total_signals": buy_signals + sell_signals,
                "agent": self.name
            }

        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = SimpleRSIAgent(period=14, oversold=30, overbought=70)

    # Test with multiple symbols
    test_symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']

    print("🤖 Simple RSI Agent Test\n" + "="*50)

    for symbol in test_symbols:
        print(f"\n📊 Analyzing {symbol}...")
        signal = agent.generate_signal(symbol)

        print(f"Signal: {signal['action']}")
        print(f"Confidence: {signal['confidence']:.2%}")

        if 'indicators' in signal['metadata']:
            indicators = signal['metadata']['indicators']
            print(f"RSI: {indicators.get('rsi', 'N/A'):.2f}")
            print(f"Price: ${indicators.get('price', 'N/A'):.2f}")

        print(f"Reasoning: {signal['metadata'].get('reasoning', 'N/A')}")

    # Run a simple backtest
    print("\n📈 Running Backtest for AAPL (6 months)...")
    backtest_result = agent.backtest('AAPL', '6mo')

    if 'error' not in backtest_result:
        print(f"Total Return: {backtest_result['total_return']:.2f}%")
        print(f"Buy Signals: {backtest_result['buy_signals']}")
        print(f"Sell Signals: {backtest_result['sell_signals']}")
