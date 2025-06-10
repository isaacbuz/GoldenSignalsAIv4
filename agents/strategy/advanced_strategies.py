import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats

class AdvancedStrategies:
    """
    Unified registry and implementation for all advanced trading strategies in GoldenSignalsAI.
    Supports registration, config-driven parameterization, and runtime selection.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get_strategy(cls, name):
        return cls._registry.get(name)

    @classmethod
    def available_strategies(cls):
        return list(cls._registry.keys())

    @classmethod
    def run_strategy(cls, name, *args, **kwargs):
        strat = cls.get_strategy(name)
        if not strat:
            raise ValueError(f"Strategy '{name}' not found. Available: {cls.available_strategies()}")
        try:
            return strat(*args, **kwargs)
        except Exception as e:
            from infrastructure.error_handler import ErrorHandler
            ErrorHandler.handle_error(Exception(f"Error running strategy '{name}': {e}"))
            raise

    @staticmethod
    def moving_average_crossover(df: pd.DataFrame, short_window=20, long_window=50) -> pd.Series:
        """
        Moving Average Crossover strategy.
        
        Args:
            df (pd.DataFrame): Price data
            short_window (int): Short MA window
            long_window (int): Long MA window
        
        Returns:
            pd.Series: Trading signals
        """
        short_ma = df['Close'].rolling(window=short_window).mean()
        long_ma = df['Close'].rolling(window=long_window).mean()
        signals = np.where(short_ma > long_ma, 1, 0)
        return pd.Series(signals, index=df.index)

    @staticmethod
    def rsi_strategy(df: pd.DataFrame, lower=30, upper=70) -> pd.Series:
        """
        RSI-based trading strategy.
        
        Args:
            df (pd.DataFrame): Price data
            lower (int): Lower RSI threshold
            upper (int): Upper RSI threshold
        
        Returns:
            pd.Series: Trading signals
        """
        rsi = talib.RSI(df['Close'].values, timeperiod=14)
        signals = np.where(rsi < lower, 1, np.where(rsi > upper, -1, 0))
        return pd.Series(signals, index=df.index)

    @staticmethod
    def rsi(df: pd.DataFrame, period=14) -> pd.Series:
        """
        Calculate RSI.
        
        Args:
            df (pd.DataFrame): Price data
            period (int): RSI calculation period
        
        Returns:
            pd.Series: RSI values
        """
        return pd.Series(talib.RSI(df['Close'].values, timeperiod=period), index=df.index)

    @staticmethod
    def macd(df: pd.DataFrame, span_short=12, span_long=26, span_signal=9) -> pd.DataFrame:
        """
        Calculate MACD.
        
        Args:
            df (pd.DataFrame): Price data
            span_short (int): Short MACD span
            span_long (int): Long MACD span
            span_signal (int): Signal MACD span
        
        Returns:
            pd.DataFrame: MACD values
        """
        macd, macdsignal, macdhist = talib.MACD(df['Close'].values, fastperiod=span_short, slowperiod=span_long, signalperiod=span_signal)
        return pd.DataFrame({'MACD': macd, 'Signal': macdsignal, 'Hist': macdhist}, index=df.index)

    @staticmethod
    def pairs_trading(asset1_prices: np.ndarray, asset2_prices: np.ndarray, window: int = 30) -> Dict[str, Any]:
        """
        Implement pairs trading strategy.
        
        Args:
            asset1_prices (np.ndarray): Prices of first asset
            asset2_prices (np.ndarray): Prices of second asset
            window (int): Rolling window for correlation and spread calculation
        
        Returns:
            Dict[str, Any]: Trading signals and metrics
        """
        spread = asset1_prices - asset2_prices
        z_score = stats.zscore(spread)
        
        signals = np.zeros_like(z_score)
        signals[z_score > 2] = -1  # Sell spread when z-score is high
        signals[z_score < -2] = 1   # Buy spread when z-score is low
        
        return {
            'signals': signals,
            'z_score': z_score,
            'strategy': 'pairs_trading'
        }

    @staticmethod
    def momentum_strategy(prices: np.ndarray, window: int = 14) -> Dict[str, Any]:
        """
        Implement momentum trading strategy using RSI.
        
        Args:
            prices (np.ndarray): Asset prices
            window (int): RSI calculation window
        
        Returns:
            Dict[str, Any]: Trading signals and RSI
        """
        rsi = talib.RSI(prices, timeperiod=window)
        
        signals = np.zeros_like(rsi)
        signals[rsi < 30] = 1   # Buy signal (oversold)
        signals[rsi > 70] = -1  # Sell signal (overbought)
        
        return {
            'signals': signals,
            'rsi': rsi,
            'strategy': 'momentum'
        }

    @staticmethod
    def volatility_breakout(prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
        """
        Volatility breakout strategy.
        
        Args:
            prices (np.ndarray): Asset prices
            window (int): Volatility calculation window
        
        Returns:
            Dict[str, Any]: Trading signals and volatility metrics
        """
        volatility = talib.STDDEV(prices, timeperiod=window)
        
        upper_band = talib.BBANDS(prices, timeperiod=window)[0]
        lower_band = talib.BBANDS(prices, timeperiod=window)[2]
        
        signals = np.zeros_like(prices)
        signals[prices > upper_band] = 1   # Breakout buy
        signals[prices < lower_band] = -1  # Breakout sell
        
        return {
            'signals': signals,
            'volatility': volatility,
            'strategy': 'volatility_breakout'
        }

    @staticmethod
    def pattern_recognition(prices: np.ndarray) -> Dict[str, Any]:
        """
        Identify chart patterns using technical analysis.
        
        Args:
            prices (np.ndarray): Asset prices
        
        Returns:
            Dict[str, Any]: Detected patterns and signals
        """
        patterns = {
            'hammer': talib.CDLHAMMER(prices),
            'inverted_hammer': talib.CDLINVERTEDHAMMER(prices),
            'morning_star': talib.CDLMORNINGSTAR(prices),
            'three_white_soldiers': talib.CDL3WHITESOLDIERS(prices)
        }
        
        signals = np.zeros_like(prices)
        for pattern_name, pattern_signals in patterns.items():
            signals[pattern_signals > 0] = 1   # Bullish patterns
            signals[pattern_signals < 0] = -1  # Bearish patterns
        
        return {
            'signals': signals,
            'patterns': patterns,
            'strategy': 'pattern_recognition'
        }

    @staticmethod
    def adaptive_strategy(prices: np.ndarray, window: int = 50) -> Dict[str, Any]:
        """
        Adaptive strategy that dynamically adjusts parameters.
        
        Args:
            prices (np.ndarray): Asset prices
            window (int): Adaptive window
        
        Returns:
            Dict[str, Any]: Adaptive trading signals
        """
        short_ma = talib.SMA(prices, timeperiod=window//2)
        long_ma = talib.SMA(prices, timeperiod=window)
        
        volatility = talib.STDDEV(prices, timeperiod=window)
        dynamic_threshold = volatility.mean()
        
        signals = np.zeros_like(prices)
        signals[short_ma > long_ma] = 1    # Bullish crossover
        signals[short_ma < long_ma] = -1   # Bearish crossover
        
        signals[volatility > dynamic_threshold] *= 0.5
        
        return {
            'signals': signals,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'strategy': 'adaptive'
        }

    @staticmethod
    def machine_learning_strategy(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Machine learning-based trading strategy.
        
        Args:
            features (np.ndarray): Input features
            labels (np.ndarray): Training labels
        
        Returns:
            Dict[str, Any]: ML trading strategy results
        """
        class MLTradingModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3),  # 3 output classes: buy, hold, sell
                    nn.Softmax(dim=1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Preprocess data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Prepare PyTorch tensors
        X = torch.FloatTensor(scaled_features)
        y = torch.LongTensor(labels)
        
        # Initialize model
        model = MLTradingModel(input_size=features.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Predict signals
        with torch.no_grad():
            predictions = model(X).argmax(dim=1).numpy()
        
        return {
            'signals': predictions,
            'model': model,
            'strategy': 'machine_learning'
        }
    
    @classmethod
    def combine_strategies(
        cls, 
        price_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Combine multiple strategies for robust trading signals.
        
        Args:
            price_data (np.ndarray): Historical price data
        
        Returns:
            Dict[str, Any]: Aggregated trading strategy
        """
        # Apply multiple strategies
        strategies = [
            cls.momentum_strategy(price_data),
            cls.volatility_breakout(price_data),
            cls.pattern_recognition(price_data),
            cls.adaptive_strategy(price_data)
        ]
        
        # Aggregate signals
        combined_signals = np.zeros_like(price_data)
        strategy_weights = [0.3, 0.2, 0.3, 0.2]  # Adjustable weights
        
        for strategy, weight in zip(strategies, strategy_weights):
            combined_signals += strategy['signals'] * weight
        
        # Normalize and discretize signals
        normalized_signals = np.sign(combined_signals)
        
        return {
            'combined_signals': normalized_signals,
            'individual_strategies': strategies,
            'strategy': 'multi_strategy_ensemble'
        }

def main():
    """
    Demonstration of advanced trading strategies.
    """
    # Simulate price data
    np.random.seed(42)
    price_data = np.cumsum(np.random.normal(0, 1, 1000))
    
    # Demonstrate strategy usage
    strategies = AdvancedTradingStrategies
    
    # Run individual strategies
    momentum = strategies.momentum_strategy(price_data)
    volatility = strategies.volatility_breakout(price_data)
    patterns = strategies.pattern_recognition(price_data)
    adaptive = strategies.adaptive_strategy(price_data)
    
    # Combine strategies
    combined_strategy = strategies.combine_strategies(price_data)
    
    print("Momentum Strategy Signals:", momentum['signals'][:10])
    print("Volatility Breakout Signals:", volatility['signals'][:10])
    print("Pattern Recognition Signals:", patterns['signals'][:10])
    print("Adaptive Strategy Signals:", adaptive['signals'][:10])
    print("Combined Strategy Signals:", combined_strategy['combined_signals'][:10])

# Register strategies after class definition
AdvancedStrategies.register('moving_average_crossover')(AdvancedStrategies.moving_average_crossover)
AdvancedStrategies.register('rsi_strategy')(AdvancedStrategies.rsi_strategy)
AdvancedStrategies.register('rsi')(AdvancedStrategies.rsi)
AdvancedStrategies.register('macd')(AdvancedStrategies.macd)
AdvancedStrategies.register('pairs_trading')(AdvancedStrategies.pairs_trading)
AdvancedStrategies.register('momentum')(AdvancedStrategies.momentum_strategy)
AdvancedStrategies.register('volatility_breakout')(AdvancedStrategies.volatility_breakout)
AdvancedStrategies.register('pattern_recognition')(AdvancedStrategies.pattern_recognition)
AdvancedStrategies.register('adaptive')(AdvancedStrategies.adaptive_strategy)
AdvancedStrategies.register('machine_learning')(AdvancedStrategies.machine_learning_strategy)

if __name__ == '__main__':
    main()
