import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats

class AdvancedTradingStrategies:
    """
    Comprehensive collection of advanced trading strategies 
    and technical analysis indicators.
    """
    
    @staticmethod
    def pairs_trading(
        asset1_prices: np.ndarray, 
        asset2_prices: np.ndarray, 
        window: int = 30
    ) -> Dict[str, Any]:
        """
        Implement pairs trading strategy.
        
        Args:
            asset1_prices (np.ndarray): Prices of first asset
            asset2_prices (np.ndarray): Prices of second asset
            window (int): Rolling window for correlation and spread calculation
        
        Returns:
            Dict[str, Any]: Trading signals and metrics
        """
        # Calculate spread and z-score
        spread = asset1_prices - asset2_prices
        z_score = stats.zscore(spread)
        
        # Identify trading opportunities
        signals = np.zeros_like(z_score)
        signals[z_score > 2] = -1  # Sell spread when z-score is high
        signals[z_score < -2] = 1   # Buy spread when z-score is low
        
        return {
            'signals': signals,
            'z_score': z_score,
            'strategy': 'pairs_trading'
        }
    
    @staticmethod
    def momentum_strategy(
        prices: np.ndarray, 
        window: int = 14
    ) -> Dict[str, Any]:
        """
        Implement momentum trading strategy using RSI.
        
        Args:
            prices (np.ndarray): Asset prices
            window (int): RSI calculation window
        
        Returns:
            Dict[str, Any]: Trading signals and RSI
        """
        # Calculate RSI
        rsi = talib.RSI(prices, timeperiod=window)
        
        # Generate signals
        signals = np.zeros_like(rsi)
        signals[rsi < 30] = 1   # Buy signal (oversold)
        signals[rsi > 70] = -1  # Sell signal (overbought)
        
        return {
            'signals': signals,
            'rsi': rsi,
            'strategy': 'momentum'
        }
    
    @staticmethod
    def volatility_breakout(
        prices: np.ndarray, 
        window: int = 20
    ) -> Dict[str, Any]:
        """
        Volatility breakout strategy.
        
        Args:
            prices (np.ndarray): Asset prices
            window (int): Volatility calculation window
        
        Returns:
            Dict[str, Any]: Trading signals and volatility metrics
        """
        # Calculate volatility
        volatility = talib.STDDEV(prices, timeperiod=window)
        
        # Calculate Bollinger Bands
        upper_band = talib.BBANDS(prices, timeperiod=window)[0]
        lower_band = talib.BBANDS(prices, timeperiod=window)[2]
        
        # Generate signals
        signals = np.zeros_like(prices)
        signals[prices > upper_band] = 1   # Breakout buy
        signals[prices < lower_band] = -1  # Breakout sell
        
        return {
            'signals': signals,
            'volatility': volatility,
            'strategy': 'volatility_breakout'
        }
    
    @staticmethod
    def pattern_recognition(
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Identify chart patterns using technical analysis.
        
        Args:
            prices (np.ndarray): Asset prices
        
        Returns:
            Dict[str, Any]: Detected patterns and signals
        """
        # Detect various chart patterns
        patterns = {
            'hammer': talib.CDLHAMMER(prices),
            'inverted_hammer': talib.CDLINVERTEDHAMMER(prices),
            'morning_star': talib.CDLMORNINGSTAR(prices),
            'three_white_soldiers': talib.CDL3WHITESOLDIERS(prices)
        }
        
        # Aggregate signals
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
    def adaptive_strategy(
        prices: np.ndarray, 
        window: int = 50
    ) -> Dict[str, Any]:
        """
        Adaptive strategy that dynamically adjusts parameters.
        
        Args:
            prices (np.ndarray): Asset prices
            window (int): Adaptive window
        
        Returns:
            Dict[str, Any]: Adaptive trading signals
        """
        # Calculate moving averages
        short_ma = talib.SMA(prices, timeperiod=window//2)
        long_ma = talib.SMA(prices, timeperiod=window)
        
        # Dynamic threshold based on recent volatility
        volatility = talib.STDDEV(prices, timeperiod=window)
        dynamic_threshold = volatility.mean()
        
        # Generate signals
        signals = np.zeros_like(prices)
        signals[short_ma > long_ma] = 1    # Bullish crossover
        signals[short_ma < long_ma] = -1   # Bearish crossover
        
        # Apply dynamic volatility filter
        signals[volatility > dynamic_threshold] *= 0.5
        
        return {
            'signals': signals,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'strategy': 'adaptive'
        }
    
    @staticmethod
    def machine_learning_strategy(
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, Any]:
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

if __name__ == '__main__':
    main()
